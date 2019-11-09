# coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from sklearn.metrics import accuracy_score

import numpy as np
import tensorflow as tf
import os
import random
import sys
import time
import math

# load keras modules
import keras
from keras import backend as K
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras.models import load_model

# attack related modules
from setup_mnist import MNIST
from attack_local_ensemble import LinfPGDAttack
from mnist_models import mnist_models
from setup_codec import CODEC
from autozoom_attack_graph import AutoZOOM
from mnist_robust_models import MadryModel, PercyModel, ZicoModel
from attack_utils import autozoom_attack, nes_attack
from utils import keras_model_wrapper, generate_attack_inputs, \
	compute_cw_loss, select_next_seed, mixup_data, local_attack_in_batches

import argparse

def main(args):
	if args["model_type"] == "normal":
		load_robust = False
	else:
		load_robust = True
	# Set TF random seed to improve reproducibility
	# tf.set_random_seed(args["seed"])
	data  = MNIST()
	if not hasattr(K, "tf"):
		raise RuntimeError("This tutorial requires keras to be configured"
						" to use the TensorFlow backend.")

	if keras.backend.image_dim_ordering() != 'tf':
		keras.backend.set_image_dim_ordering('tf')
		print("INFO: '~/.keras/keras.json' sets 'image_dim_ordering' to "
			"'th', temporarily setting to 'tf'")

	# Create TF session and set as Keras backend session
	sess = tf.Session()
	keras.backend.set_session(sess)

	# # Get MNIST test data
	x_test, y_test = MNIST().test_data, MNIST().test_labels

	all_trans_rate_ls=[] # store transfer rate of all seeds (evaluated on independent test set)
	remain_trans_rate_ls = [] # store transfer rate of remaining seeds, used only in local model fine-tuning

	# Define input TF placeholder
	x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
	y = tf.placeholder(tf.float32, shape=(None, 10))
	class_num = 10
	image_size = 28
	num_channels = 1

	################## load target model ###############################	
	if not load_robust:
		target_model_name = 'mnist'
		target_model = mnist_models(sess,0,use_softmax=True,x = x, y = y,\
			load_existing=True,model_name=target_model_name)
		accuracy = target_model.calcu_acc(x_test,y_test,batch_size = 1000)
		print('Test accuracy of target model {}: {:.4f}'.format(target_model_name,accuracy))
	else:
		if args["robust_type"] == "madry":
			target_model_name = 'madry_robust'
			model_dir = "MNIST_models/madry_robust_model" 
			target_model = MadryModel(sess,model_dir = model_dir,bias = 0.5)
		elif args["robust_type"] == "zico":
			target_model_name = 'zico_robust'
			model_dir = "MNIST_models/zico_robust_model/mnist.pth"  
			target_model = ZicoModel(model_dir = model_dir,bias = 0.5)
		elif args["robust_type"] == "percy":
			target_model_name = 'percy_robust'
			model_dir = "MNIST_models/percy_robust_model/sdp_train/" 
			target_model = PercyModel(sess,model_dir = model_dir,bias = 0.5)
		else:
			raise NotImplementedError
		pred_class = target_model.pred_class(x_test)
		print('Test accuracy of target robust model :{:.4f}'.format(np.sum(pred_class == np.argmax(y_test,axis = 1))/len(x_test))) 
	###################### end of loading target model ################## 

	if args["attack_method"] == "autozoom":
		# setup the autoencoders
		# codec = CODEC(image_size, num_channels, args["compress_mode"], use_tanh=False)
		codec = 0
		args["img_resize"] = 14
		codec_dir = 'MNIST_models/mnist_autoencoder/'
		encoder = load_model(codec_dir + 'whole_mnist_encoder.h5')
		decoder = load_model(codec_dir + 'whole_mnist_decoder.h5')

		encode_img = encoder.predict(data.test_data[100:101])
		decode_img = decoder.predict(encode_img)
		diff_img = (decode_img - data.test_data[100:101])
		diff_mse = np.mean(diff_img.reshape(-1)**2)

		print("[Info][AE] MSE:{:.4f}".format(diff_mse))
		encode_img = encoder.predict(data.test_data[0:1])
		decode_img = decoder.predict(encode_img)
		diff_img = (decode_img - data.test_data[0:1])
		diff_mse = np.mean(diff_img.reshape(-1)**2)
		print("[Info][AE] MSE:{:.4f}".format(diff_mse))

	#load local models or define the architecture
	local_model_names = ['modelA','modelB','modelC', 'modelD', 'modelE', 'modelF']

	if args["attack_method"] == "autozoom":
		# define black-box model graph 
		blackbox_attack = AutoZOOM(sess, target_model, args, decoder, codec,
				num_channels,image_size,class_num) 

	nb_imgs = args["num_img"]
	# local attack related params
	clip_min = -0.5
	clip_max = 0.5
	li_eps = args["cost_threshold"]

	load_existing = True # if true, we start with pretrained local models, otherwise, start with random model
	with_local = args["with_local"] # if true, hybrid attack, otherwise, only baseline attacks
	
	if args["no_tune_local"]:
		stop_fine_tune_flag = True
	else:
		stop_fine_tune_flag = False

	if with_local:
		if load_existing:
			loc_adv = 'adv_with_tune'
		if args["no_tune_local"]:
			loc_adv = 'adv_no_tune'
	else:
		loc_adv = 'orig'
	
	# target type
	if args["attack_type"] == "targeted":
		is_targeted = True
	else:
		is_targeted = False

	### attack pretrained models or build models using substitute training ####   
	# local model training parameters
	sub_epochs = args["nb_epochs_sub"] # epcohs for sub model training or fine-tuning
	use_loc_adv_thres = args["use_loc_adv_thres"] # threshold for transfer attack success rate, only use local AEs when transfer rate is higher than this threshold
	use_loc_adv_flag = True # since we are starting from pretrained models, hybrid attack always uses local aes. If we start from random model, local ae is not reliable

	fine_tune_freq = args["fine_tune_freq"] # fine-tune the model every K images to save total model training time
	
	# store the attack input files (e.g., original image, target class)
	input_file_prefix = os.path.join(args["local_path"],target_model_name,
												args["attack_type"])
	os.system("mkdir -p {}".format(input_file_prefix)) 
	# generate the attack seeds and target class
	target_ys_one_hot,orig_images,target_ys,orig_labels,_, x_trans_inputs = \
	generate_attack_inputs(sess,target_model,x_test,y_test,class_num,nb_imgs,\
		load_imgs=args["load_imgs"],load_robust=load_robust,\
			file_path = input_file_prefix)

	# images are generated based on seed (1234), reassign 
	# the random to improve reproducibility
	random.seed(args["seed"])
	np.random.seed(args["seed"])
	tf.set_random_seed(args["seed"])

	start_points = np.copy(orig_images) # starting points for the gradient attacks, can be local AEs
	# attack statistical info
	dist_record = np.zeros(len(orig_labels),dtype = float)  
	query_num_vec = np.zeros(len(orig_labels), dtype=int)   
	success_vec = np.zeros(len(orig_labels),dtype=bool)
	adv_classes = np.zeros(len(orig_labels), dtype=int)

	# define local model graphs and do the initial training
	model_types = [0,1,2,5] # select the architecture of local models
	local_model_ls = []
	pred_ls = []
	sss = 0
	# Prepare callbacks for model saving
	save_dir = 'MNIST_models/normal_models/' 
	callbacks_ls = []

	attacked_flag = np.zeros(len(orig_labels),dtype = bool)
	################## load local models #####################
	if with_local:
		load_model_names = []
		for model_type in model_types:
			model_name = local_model_names[model_type]
			load_model_names.append(model_name)
			loc_model = mnist_models(sess,model_type,use_softmax=True,x = x, y = y,\
			load_existing=load_existing,model_name=model_name)
			pred_ls.append(loc_model.predictions)
			local_model_ls.append(loc_model)
			opt = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
			loc_model.model.compile(loss='categorical_crossentropy',
										optimizer=opt,
										metrics=['accuracy'])
			
			if args["no_save_model"]:
				if not load_existing:
					loc_model.model.fit(orig_images, orig_labels,
						batch_size=args["train_batch_size"],
						epochs=sub_epochs,
						verbose=0,
						validation_data=(x_test, y_test),
						shuffle = True) 
			else:
				if load_existing:
					filepath = save_dir + model_name + '_pretrained.h5' 
				else:
					filepath = save_dir + model_name + '.h5' 
				checkpoint = ModelCheckpoint(filepath=filepath,
											monitor='val_acc',
											verbose=0,
											save_best_only=True)

				# earlystopping = keras.callbacks.EarlyStopping(monitor='val_acc',\
				# min_delta=0, patience=3, verbose=0, mode='auto', baseline=None, \
				# restore_best_weights=False)
				# callbacks = [checkpoint, earlystopping]
				callbacks = [checkpoint]
				callbacks_ls.append(callbacks)
				if not load_existing:
					print("Train on %d data and validate on %d data" % (len(orig_labels),len(y_test)))
					loc_model.model.fit(orig_images, orig_labels,
						batch_size=args["train_batch_size"],
						epochs=sub_epochs,
						verbose=0,
						validation_data=(x_test, y_test),
						shuffle = True,
						callbacks = callbacks)  
			scores = loc_model.model.evaluate(x_test, y_test, verbose=0)
			print('Test accuracy of model {}: {:.4f}'.format(model_name,scores[1]))
			sss += 1         
		####################### end of load local models ########################
		# Define Attack Graph of PGD attack 
		local_attack_graph = LinfPGDAttack(local_model_ls,
							epsilon = li_eps, 
							k = 100,
							a = 0.01,
							random_start = True,
							loss_func = args["loss_function"],
							targeted = is_targeted,
							x = x,
							y = y)

		# store local info: a directory of random number generator is included below for the convenience in averaging the attack results
		local_info_file_prefix = os.path.join(args["local_path"],target_model_name,
												args["attack_type"],str(args["seed"]))
		os.system("mkdir -p {}".format(local_info_file_prefix)) 
		if not args["load_local_AEs"]:
			# check do the transfer check to obtain local adversarial samples
			if is_targeted:
				all_trans_rate, pred_labs, local_aes,pgd_cnt_mat, max_loss, min_loss,\
             ave_loss, max_gap, min_gap, ave_gap = local_attack_in_batches(sess,start_points[np.logical_not(attacked_flag)],\
				target_ys_one_hot[np.logical_not(attacked_flag)],eval_batch_size = 500,\
				attack_graph = local_attack_graph,model = target_model,clip_min=clip_min,clip_max=clip_max,load_robust=load_robust)
			else:
				all_trans_rate, pred_labs, local_aes,pgd_cnt_mat,max_loss, min_loss,\
             ave_loss, max_gap, min_gap, ave_gap = local_attack_in_batches(sess,start_points[np.logical_not(attacked_flag)],\
				orig_labels[np.logical_not(attacked_flag)],eval_batch_size = 500,\
				attack_graph = local_attack_graph,model = target_model,clip_min=clip_min,clip_max=clip_max,load_robust=load_robust)

			# calculate local adv loss used for scheduling experiments...
			if is_targeted:
				adv_img_loss, free_idx = compute_cw_loss(sess,target_model,local_aes,\
				target_ys_one_hot,targeted=is_targeted,load_robust=load_robust)
			else:
				adv_img_loss, free_idx = compute_cw_loss(sess,target_model,local_aes,\
				orig_labels,targeted=is_targeted,load_robust=load_robust)
			
			# calculate orig img loss for scheduling experiments
			if is_targeted:
				orig_img_loss, free_idx = compute_cw_loss(sess,target_model,orig_images,\
				target_ys_one_hot,targeted=is_targeted,load_robust=load_robust)
			else:
				orig_img_loss, free_idx = compute_cw_loss(sess,target_model,orig_images,\
				orig_labels,targeted=is_targeted,load_robust=load_robust)

			if not args["force_tune_baseline"]:
				# save local aes
				np.save(local_info_file_prefix+'/local_aes.npy',local_aes)
				# store local info of local aes and original seeds: used for scheduling seeds in batch attacks
				np.savetxt(local_info_file_prefix+'/pgd_cnt_mat.txt',pgd_cnt_mat)
				np.savetxt(local_info_file_prefix+'/orig_img_loss.txt',orig_img_loss)
				np.savetxt(local_info_file_prefix+'/adv_img_loss.txt',adv_img_loss)
				np.savetxt(local_info_file_prefix+'/ave_gap.txt',ave_gap)
		else:
			local_aes = np.load(local_info_file_prefix+'/local_aes.npy')
			if is_targeted:
				tmp_labels = target_ys_one_hot
			else:
				tmp_labels = orig_labels
			pred_labs = np.argmax(target_model.predict_prob(np.array(local_aes)),axis=1)
			print('correct number',np.sum(pred_labs == np.argmax(tmp_labels,axis=1)))
			all_trans_rate = accuracy_score(np.argmax(tmp_labels,axis=1), pred_labs)
		if not is_targeted:
			all_trans_rate = 1 - all_trans_rate
		print('** Transfer Rate: **' + str(all_trans_rate))  

		# independent test set for checking transferability: for experiment purpose and does not count for query numbers
		if is_targeted:
			ind_all_trans_rate, pred_labs, _,_, _, _,\
             _, _, _, _ = local_attack_in_batches(sess,x_trans_inputs,target_ys_one_hot,eval_batch_size = 500,\
			attack_graph = local_attack_graph,model = target_model,clip_min=clip_min,clip_max=clip_max,load_robust=load_robust)
		else:
			ind_all_trans_rate, pred_labs, _,_,_, _,\
             _, _, _, _ = local_attack_in_batches(sess,x_trans_inputs,orig_labels,eval_batch_size = 500,\
			attack_graph = local_attack_graph,model = target_model,clip_min=clip_min,clip_max=clip_max,load_robust=load_robust)
		
		# record the queries spent by quering the local samples
		query_num_vec[np.logical_not(attacked_flag)] += 1
		if not is_targeted:
			ind_all_trans_rate = 1 - ind_all_trans_rate
		print('** (Independent Set) Transfer Rate: **' + str(ind_all_trans_rate))   
		all_trans_rate_ls.append(ind_all_trans_rate)
		if args["test_trans_rate_only"]:
			print("Program terminates after checking the transfer rate!")
			sys.exit(0)
		if not args["force_tune_baseline"]:
			if all_trans_rate > use_loc_adv_thres:
				print("Updated the starting points to local AEs....")
				start_points[np.logical_not(attacked_flag)] = local_aes
				use_loc_adv_flag = True

	# initial fine-tuning set obtained from querying the target model
	S = np.copy(start_points)
	S_label = target_model.predict_prob(S)
	S_label_cate = np.argmax(S_label,axis = 1)
	S_label_cate = np_utils.to_categorical(S_label_cate, class_num)

	# stores the order of images attacked (record the image index)
	candi_idx_ls = []
	pre_free_idx = []
	# these parameters are used to make sure equal number of instances from each class are selected
	# such that diversity of fine-tuning set is improved. However, it is not effective...
	per_cls_cnt = 0
	cls_order = 0
	change_limit = False
	max_lim_num = int(fine_tune_freq/class_num) 

	# store gradient black-box attack results
	out_dir_prefix = os.path.join(args["save_path"], args["attack_method"],target_model_name,
												args["attack_type"],str(args["seed"])) 
	os.system("mkdir -p {}".format(out_dir_prefix)) 

	######### main loop of hybrid attack ###########
	for itr in range(len(orig_labels)):
		print("#------------ Attack Round {} ----------------#".format(itr))
		if is_targeted:
			img_loss, free_idx = compute_cw_loss(sess,target_model,start_points,\
			target_ys_one_hot,targeted=is_targeted,load_robust=load_robust)
		else:
			img_loss, free_idx = compute_cw_loss(sess,target_model,start_points,\
			orig_labels,targeted=is_targeted,load_robust=load_robust)

		free_idx_diff = list(set(free_idx) - set(pre_free_idx))
		print("new free_idx found:",free_idx_diff)
		if len(free_idx_diff) > 0:
			candi_idx_ls.extend(free_idx_diff)
		pre_free_idx = free_idx
		
		if with_local:
			if len(free_idx)>0:
				# free attacks are found
				attacked_flag[free_idx] = 1 
				success_vec[free_idx] = 1
				# update the distance and adv class
				if args['dist_metric'] == 'l2':
					dist = np.sum((start_points[free_idx]-orig_images[free_idx])**2,axis = (1,2,3))**.5
				elif args['dist_metric'] == 'li':
					dist = np.amax(np.abs(start_points[free_idx] - orig_images[free_idx]),axis = (1,2,3))
				adv_class = target_model.pred_class(start_points[free_idx])
				adv_classes[free_idx]= adv_class
				dist_record[free_idx] = dist 
				if np.amax(dist) >= args["cost_threshold"] + args["cost_threshold"]/10:
					print("there are some problems in setting the perturbation distance!")
					sys.exit(0)
		print("Number of Unattacked Seeds: ",np.sum(np.logical_not(attacked_flag)))
		if attacked_flag.all():
			# early stop when seeds are sucessfully attacked
			break

		if args["sort_metric"] == "min":
			img_loss[attacked_flag] = 1e10
		elif args["sort_metric"] == "max":
			img_loss[attacked_flag] = -1e10
		candi_idx, per_cls_cnt, cls_order,change_limit,max_lim_num = select_next_seed(img_loss,attacked_flag,args["sort_metric"],\
		args["by_class"],fine_tune_freq,class_num,per_cls_cnt,cls_order,change_limit,max_lim_num)

		print(candi_idx)
		candi_idx_ls.append(candi_idx)
		input_img = start_points[candi_idx:candi_idx+1]
		
		if args["attack_method"] == "autozoom":
			# encoder performance check
			encode_img = encoder.predict(input_img)
			decode_img = decoder.predict(encode_img)
			diff_img = (decode_img - input_img)
			diff_mse = np.mean(diff_img.reshape(-1)**2)
		else:
			diff_mse = 0.0
		print("[Info][Start]: test_index:{}, true label:{}, target label:{}, MSE:{}".format(candi_idx, np.argmax(orig_labels[candi_idx]),\
			np.argmax(target_ys_one_hot[candi_idx]),diff_mse))
		################## BEGIN: bbox attacks ############################
		if args["attack_method"] == "autozoom":
			if is_targeted:
				x_s, ae, query_num = autozoom_attack(blackbox_attack,input_img,orig_images[candi_idx:candi_idx+1],target_ys_one_hot[candi_idx])
			else:
				x_s, ae, query_num = autozoom_attack(blackbox_attack,input_img,orig_images[candi_idx:candi_idx+1],orig_labels[candi_idx])
		else:
			if is_targeted:
				x_s, query_num, ae = nes_attack(args,target_model,input_img,orig_images[candi_idx:candi_idx+1],np.argmax(target_ys_one_hot[candi_idx]),\
				lower = clip_min, upper = clip_max)
			else:
				x_s, query_num, ae = nes_attack(args,target_model,input_img,orig_images[candi_idx:candi_idx+1],np.argmax(orig_labels[candi_idx]),\
				lower = clip_min, upper = clip_max)
			x_s = np.squeeze(np.array(x_s),axis = 1)
		################## END: bbox attacks ############################

		attacked_flag[candi_idx] = 1
		# fill the query info, etc
		if len(ae.shape) == 3:
			ae = np.expand_dims(ae, axis=0)
		if args['dist_metric'] == 'l2':
			dist = np.sum((ae-orig_images[candi_idx])**2)**.5
		elif args['dist_metric'] == 'li':
			dist = np.amax(np.abs(ae-orig_images[candi_idx]))
		adv_class = target_model.pred_class(ae)
		adv_classes[candi_idx] = adv_class
		query_num_vec[candi_idx] += query_num 

		if dist >= args["cost_threshold"] + args["cost_threshold"]/10 or math.isnan(dist):
			print("the distance is not optimized properly")
			sys.exit(0)

		if is_targeted:
			if adv_class == np.argmax(target_ys_one_hot[candi_idx]):
				success_vec[candi_idx] = 1
		else:
			if adv_class != np.argmax(orig_labels[candi_idx]):
				success_vec[candi_idx] = 1
		
		if attacked_flag.all():
			print("Early termination because all seeds are successfully attacked!")
			break
		##############################################################
		## Starts the section of substitute training and local advs ##
		##############################################################
		if with_local:
			if not stop_fine_tune_flag:
				# augment the local model training data with target model labels
				S = np.concatenate((S, np.array(x_s)), axis=0)        
				S_label_add = target_model.predict_prob(np.array(x_s))
				S_label_add_cate = np.argmax(S_label_add,axis = 1)
				S_label_add_cate = np_utils.to_categorical(S_label_add_cate, class_num)
				S_label_cate = np.concatenate((S_label_cate, S_label_add_cate), axis=0)
				S_label = np.concatenate((S_label, S_label_add), axis=0)            
				# fine-tune the local models
				if itr % fine_tune_freq == 0 and itr != 0:
					if len(S_label) > args["train_inst_lim"]:
						curr_len = len(S_label)
						rand_idx = np.random.choice(len(S_label),args["train_inst_lim"],replace = False)
						S = S[rand_idx]
						S_label = S_label[rand_idx]
						S_label_cate = S_label_cate[rand_idx]  
						print("current num: %d, max train instance limit %d is reached, performed random sampling to get %d samples!" % (curr_len,len(S_label),len(rand_idx)))  
					
					print("Train on %d data and validate on %d data" % (len(S_label),len(y_test)))
					sss = 0
					for loc_model in local_model_ls:
						# model_name = local_model_names[sss]
						model_name = load_model_names[sss]
						if args["no_save_model"]:
							loc_model.model.fit(S, S_label,
							batch_size=args["train_batch_size"],
							epochs=sub_epochs,
							verbose=0,
							validation_data=(x_test, y_test),
							shuffle = True)  
						else:
							callbacks = callbacks_ls[sss]
							loc_model.model.fit(S, S_label,
								batch_size=args["train_batch_size"],
								epochs=sub_epochs,
								verbose=0,
								validation_data=(x_test, y_test),
								shuffle = True,
								callbacks = callbacks)     
						scores = loc_model.model.evaluate(x_test, y_test, verbose=0)
						print('Test accuracy of model {}: {:.4f}'.format(model_name,scores[1]))
						sss += 1

					if not attacked_flag.all():
						# first check for not attacked seeds
						if is_targeted:
							remain_trans_rate, pred_labs, remain_local_aes,_, _, _,\
             				_, _, _, _ = local_attack_in_batches(sess,orig_images[np.logical_not(attacked_flag)],\
							target_ys_one_hot[np.logical_not(attacked_flag)],eval_batch_size = 500,\
							attack_graph = local_attack_graph,model = target_model,clip_min=clip_min,clip_max=clip_max,load_robust=load_robust)
						else:
							remain_trans_rate, pred_labs, remain_local_aes,_,_, _,\
             				_, _, _, _ = local_attack_in_batches(sess,orig_images[np.logical_not(attacked_flag)],\
							orig_labels[np.logical_not(attacked_flag)],eval_batch_size = 500,\
							attack_graph = local_attack_graph,model = target_model,clip_min=clip_min,clip_max=clip_max,load_robust=load_robust)
						if not is_targeted:
							remain_trans_rate = 1 - remain_trans_rate
						print('<<Ramaining Seed Transfer Rate>>:**' + str(remain_trans_rate))
						# if transfer rate is higher than threshold, use local advs as starting points
						if remain_trans_rate <=0 and use_loc_adv_flag:
							print("No improvement from substitue training, stop fine-tuning!")
							stop_fine_tune_flag = False

						# transfer rate check with independent test examples
						if is_targeted:
							all_trans_rate, pred_labs, _,_, _, _,\
             			_, _, _, _ = local_attack_in_batches(sess,x_trans_inputs,target_ys_one_hot,eval_batch_size = 500,\
							attack_graph = local_attack_graph,model = target_model,clip_min=clip_min,clip_max=clip_max,load_robust=load_robust)
						else:
							all_trans_rate, pred_labs, _,_, _, _,\
             			_, _, _, _ = local_attack_in_batches(sess,x_trans_inputs,orig_labels,eval_batch_size = 500,\
							attack_graph = local_attack_graph,model = target_model,clip_min=clip_min,clip_max=clip_max,load_robust=load_robust)
						if not is_targeted:
							all_trans_rate = 1 - all_trans_rate
						print('<<Overall Transfer Rate>>: **'+str(all_trans_rate))
						
						# if trans rate is not high enough, still start from orig seed; start from loc adv only 
						# when trans rate is high enough
						if not args["force_tune_baseline"]:
							if not use_loc_adv_flag:
								if remain_trans_rate > use_loc_adv_thres: 
									use_loc_adv_flag = True
									print("Updated the starting points....")
									start_points[np.logical_not(attacked_flag)] = remain_local_aes
								# record the queries spent on checking newly generated loc advs
								query_num_vec += 1
							else:
								print("Updated the starting points....")
								start_points[np.logical_not(attacked_flag)] = remain_local_aes
								# record the queries spent on checking newly generated loc advs
								query_num_vec[np.logical_not(attacked_flag)] += 1
						remain_trans_rate_ls.append(remain_trans_rate)
						all_trans_rate_ls.append(all_trans_rate)
			np.set_printoptions(precision=4)
			print("all_trans_rate:")
			print(all_trans_rate_ls)
			print("remain_trans_rate")
			print(remain_trans_rate_ls)

	# save the query information of all classes
	if not args["no_save_text"]:
		save_name_file = os.path.join(out_dir_prefix,"{}_num_queries.txt".format(loc_adv))
		np.savetxt(save_name_file, query_num_vec,fmt='%d',delimiter=' ')
		save_name_file = os.path.join(out_dir_prefix,"{}_success_flags.txt".format(loc_adv))
		np.savetxt(save_name_file, success_vec,fmt='%d',delimiter=' ')

if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	## general parameters ##
	parser.add_argument("-n", "--num_img", type=int, default=100, help = "number of test images to attack")
	parser.add_argument("--cost_threshold", type=float, default=0.3, help="Feature perturbation limit (L_inf)")
	parser.add_argument("--attack_type", default="targeted", choices=["targeted", "untargeted"], help="the type of attack")
	parser.add_argument("--seed", type=int, default=1234, help="random seed")
	parser.add_argument("--dist_metric", default="li", choices=["l2", "li"], help="norm ball for attacks")
	parser.add_argument("--no_save_img", action='store_true',help="Do not save images")
	parser.add_argument("--no_save_text", action='store_true',help="Do not save attack results")
	parser.add_argument("-mt","--model_type", default="normal", choices=["normal", "robust"], help="the type of model used for attack")
	parser.add_argument("--with_local",action = "store_true",help = "Conduct hybrid attacks")
	parser.add_argument("--load_existing",action = "store_true",help = "start from a pretrained models")
	parser.add_argument("--load_imgs",action = "store_true",help = "directly load saved data as attack candidate seeds")
	parser.add_argument("--use_loc_adv_thres", type = float, default = 0.0, help = "transfer rate threshold where targeted attack is more reliable...")  
	parser.add_argument("--fine_tune_freq", type = int, default = 50, help = "frequency to check the transfer rates")  
	parser.add_argument("--nb_epochs_sub", type = int, default = 10, help = "number of epcosh to train models") 
	parser.add_argument("--no_tune_local",action = "store_true",help = "load the pretrained model without any interactions")
	parser.add_argument("--sort_metric",default="random", choices=["max", "min","random"], help="the instance selection criteria of zoo attacks (sorted order of loss function)")
	parser.add_argument("--by_class",action = "store_true",help = "generate instances from each classes")
	parser.add_argument("--no_save_model",action = "store_true",help = "if no, local models will not be saved")
	parser.add_argument("--robust_type", default="madry", choices=["madry", "percy", "zico"], help="robust models to be attacked, only used when attacking robust models")
	parser.add_argument("--attack_method", default="autozoom", choices=["autozoom", "nes"], help="black-box attack method")
	parser.add_argument("--train_inst_lim",type = int, default = 60000,help="maximum limit on fine-tuning samples that can be obtained...")
	parser.add_argument("--train_batch_size",type = int, default = 128, help = "size of training batches")
	parser.add_argument("-s", "--save_path", default="Results", help="the path to save the black-box attack results")
	parser.add_argument("--local_path", default="local_info", help="the path to save the local attack results")
	parser.add_argument("--loss_function", default="cw", choices=["xent", "cw"], help="loss function for attacking local models")
	parser.add_argument("--load_local_AEs",action = "store_true",help = "load local adversarial examples")
	parser.add_argument("--force_tune_baseline", action='store_true',help="local models are still tuned even running baseline attack")
	parser.add_argument("--test_trans_rate_only", action='store_true',help="just check the transfer rate, terminate after that")

	#parameters used for SDP based certifiable model (https://arxiv.org/pdf/1801.09344.pdf)
	parser.add_argument("--num_classes", type=int, default=10, help="number of classes")
	parser.add_argument("--num_hidden", type=int, default=500, help="number of hidden units")
	parser.add_argument("--dimension", type=int, default=784, help="dataset dimension, now only MNIST")
	parser.add_argument("--save_weight", default='sdp_train/sdp', help="path to save the model weights")

	# parameters for the autozoom attack
	parser.add_argument("-b", "--batch_size", type=int, default=1, help="the batch size for autozoom attack")
	parser.add_argument("-c", "--init_const", type=float, default=1, help="the initial setting of the constant lambda of autozoom attack")
	parser.add_argument("-m", "--max_iterations", type=int, default=0, help = "set 0 to use default value")
	parser.add_argument("-p", "--print_every", type=int, default=100, help="print information every PRINT_EVERY iterations")
	parser.add_argument("--confidence", default=0, type=float, help="the attack confidence")
	parser.add_argument("--codec_prefix", default=None, help="the coedec prefix, load the default codec is not set")
	parser.add_argument("--num_rand_vec", type=int, default=1, help="the number of random vector for post success iteration")
	parser.add_argument("--img_offset", type=int, default=0, help="the offset of the image index when getting attack data")
	parser.add_argument("--img_resize", default=None, type=int, help = "this option only works for ATTACK METHOD zoo and zoo_rv")
	parser.add_argument("--switch_iterations", type=int, default=1000, help="the iteration number for dynamic switching")
	parser.add_argument("--compress_mode", type=int, default=None, help="specify the compress mode if autoencoder is used")

	#parameters used for nes attack:
	parser.add_argument('--samples_per_draw', type=int, default=50)
	parser.add_argument('--nes_batch_size', type=int, default=50)
	parser.add_argument('--sigma', type=float, default=1e-3)
	parser.add_argument('--learning-rate', type=float, default=0.01)
	parser.add_argument('--momentum', type=float, default=0.9)
	parser.add_argument('--max-queries', type=int, default=4000)
	parser.add_argument('--plateau-drop', type=float, default=2.0) 
	parser.add_argument('--min-lr-ratio', type=int, default=200)
	parser.add_argument('--plateau-length', type=int, default=5)
	parser.add_argument('--max_lr', type=float, default=1e-2)
	parser.add_argument('--min_lr', type=float, default=5e-5)

	args = vars(parser.parse_args())

	# autozoom custom settings based on dataset and attack method
	if args["max_iterations"] == 0:
		args["max_iterations"] = 2000
	args["use_tanh"] = False
	if args["codec_prefix"] is None:
		args["codec_prefix"] = "codec/mnist_1"
	args["lr"] = 1e-2
	args["compress_mode"] = 1

	if args["img_resize"] is not None:
		if args["attack_method"] == "zoo_ae" or args["attack_method"] == "autozoom":
			print("Attack method {} cannot use option img_resize, arugment ignored".format(args["attack_method"]))
	# nes lr setup
	if args["model_type"] == "normal":
		args["min_lr"] = 5e-5
	else:
		# lr below is more effective when attacking robust models
		args["min_lr"] = 5e-3

	# setup random seed
	random.seed(args["seed"])
	np.random.seed(args["seed"])
	tf.set_random_seed(args["seed"])
	print(args)
	main(args)
