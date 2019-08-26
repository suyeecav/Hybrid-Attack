from keras.preprocessing import image

import argparse
import random
import time
import os
import math
import scipy.misc
import numpy as np
import tensorflow as tf
import keras
import sys
from sklearn.metrics import accuracy_score
from keras.utils import np_utils
from keras.models import Model, model_from_json, load_model
from cleverhans.utils_tf import model_eval

from utils import ImageNetProducer, DataSpec, load_local_model, keras_model_wrapper, compute_cw_loss
from attack_local_ensemble import LinfPGDAttack

from attack_utils import AutoZOOM


###### put imagenet evaluation dataset path here ######
INPUT_DIR = '/bigtemp/jc6ub/imagenet_tf/val/' # change here for your imagenet data path
TOT_IMAGES = 200 # total image load from dataset 


def autozoom_attack(attack_graph, input_img, orig_img, label):
	# run the zoo style attacks on selected adversarial samples and record intermediate results
	# :param: data: the image to be attacked
	ae, query_num = attack_graph.attack(input_img, label, orig_img)
	return ae, query_num

# evaluate the accuracy of models on batches
def local_attack_in_batches(sess, data, labels, eval_batch_size, attack_graph,model=None, clip_min = 0, clip_max = 1):
	# Iterate over the samples batch-by-batch
	num_eval_examples = len(data)
	num_batches = int(math.ceil(num_eval_examples / eval_batch_size))
	local_aes = [] # adv accumulator
	pgd_cnt_mat = []
	# print('Iterating over {} batches'.format(num_batches))
	for ibatch in range(num_batches):
		bstart = ibatch * eval_batch_size
		bend = min(bstart + eval_batch_size, num_eval_examples)
		# print('batch size: {}'.format(bend - bstart))
		x_batch = data[bstart:bend, :]
		y_batch = labels[bstart:bend,:]
		local_aes_batch, _, _, pgd_stp_cnt_mat = attack_graph.attack(x_batch, y_batch, sess,clip_min = clip_min,clip_max = clip_max)
		local_aes.extend(local_aes_batch)
		pgd_cnt_mat.extend(pgd_stp_cnt_mat)
	if model:
		pred_labs = np.argmax(model.predict_prob(np.array(local_aes)),axis=1)
		accuracy = accuracy_score(np.argmax(labels,axis=1), pred_labs)
	else:
		pred_labs = accuracy = 0
	return accuracy, pred_labs, np.array(local_aes), pgd_cnt_mat

def generate_attack_inputs(model, x_test, y_test, class_num, nb_imgs):
	y_probs = model.predict_prob(x_test)
	y_preds = np.argmax(y_probs, axis = 1)
	labels = np.argmax(y_test, axis = 1)
	mask = labels == y_preds
	corr_idx = np.where(labels == y_preds)[0]
	print(np.sum(mask), "out of", mask.size, "are correctly labeled, idx are:", len(x_test[mask]))
	np.random.seed(1234) # for producing the same images
	corr_cls_idx = np.random.choice(corr_idx, size=nb_imgs, replace=False)
	orig_images = x_test[corr_cls_idx]
	orig_labels = y_test[corr_cls_idx]
	target_ys = np.argmin(y_probs, axis = 1)[corr_cls_idx]
	target_ys_one_hot = np_utils.to_categorical(target_ys, class_num)

	return target_ys_one_hot, orig_images, target_ys, orig_labels

def main(args):
	tf.set_random_seed(1234) # for producing the same images

	if not hasattr(keras.backend, "tf"):
		raise RuntimeError("This tutorial requires keras to be configured"
						" to use the TensorFlow backend.")

	if keras.backend.image_dim_ordering() != 'tf':
		keras.backend.set_image_dim_ordering('tf')
		print("INFO: '~/.keras/keras.json' sets 'image_dim_ordering' to "
			"'th', temporarily setting to 'tf'")

	sess = tf.Session()
	keras.backend.set_session(sess)

	# load and preprocess dataset
	data_spec = DataSpec(batch_size=TOT_IMAGES, scale_size=256, crop_size=224, isotropic=False)
	image_producer = ImageNetProducer(
		data_path=INPUT_DIR,
		num_images=TOT_IMAGES,
		data_spec=data_spec,
		batch_size=TOT_IMAGES)

	# Define input TF placeholder
	x = tf.placeholder(tf.float32, shape=(None, 224, 224, 3))
	y = tf.placeholder(tf.float32, shape=(None, 1000))
	class_num = 1000

	# load target model and produce data
	# model = preprocess layer + pretrained model
	from keras.applications.densenet import DenseNet121
	from keras.applications.densenet import preprocess_input
	pretrained_model = DenseNet121(weights='imagenet')
	image_producer.startover()
	target_model = keras_model_wrapper(pretrained_model, preprocess_input, x = x,y = y)
	images, label = None, None
	for (indices, label, names, images) in image_producer.batches(sess):
		images = np.array(images)
		label = np_utils.to_categorical(np.array(label), class_num)
	assert type(images) == np.ndarray, type(images)
	assert type(label) == np.ndarray, type(images)

	accuracy = model_eval(sess, x, y, target_model.predictions, images, label, args= {'batch_size': 32})
	print('Test accuracy of wrapped target model:{:.4f}'.format(accuracy))
	
	# data information
	x_test, y_test = images, label # x_test [0, 255]
	print(images.shape, len(label))
	print(np.min(x_test), np.max(x_test))


	# local attack specific parameters 
	clip_min = 0.0
	clip_max = 255.0
	nb_imgs = args['num_img']
	li_eps = 12.0
	targeted_true = True if args['attack_type'] == 'targeted' else False
	k = 30 # pgd iteration
	a = 2.55 # pgd step size

	# Test the accuracy of targeted attacks, need to redefine the attack graph
	target_ys_one_hot, orig_images, target_ys, orig_labels = generate_attack_inputs(target_model, x_test,y_test, class_num, nb_imgs)

	# Set random seed to improve reproducibility
	tf.set_random_seed(args["seed"])
	np.random.seed(args["seed"])

	# test whether adversarial examples exsit, if no, generate it, otherwise, load it.
	prefix = "Results"
	prefix = os.path.join(prefix, str(args["seed"]))

	if not os.path.exists(prefix): # no history info
		# load local models or define the architecture
		local_model_types = ['VGG16', 'VGG19', 'resnet50'] # 'VGG16', 'VGG19', 'resnet50']
		local_model_ls = []
		pred_ls = []
		for model_type in local_model_types:
			pretrained_model, preprocess_input_func = load_local_model(model_type)
			local_model = keras_model_wrapper(pretrained_model, preprocess_input_func, x = x,y = y)
			accuracy = model_eval(sess, x, y, local_model.predictions, images, label, args= {'batch_size': 32})
			# assert accuracy >= 0.5, 'Error: low accuracy of local model'
			print('Test accuracy of model {}: {:.4f}'.format(model_type, accuracy))
			local_model_ls.append(local_model)
			pred_ls.append(local_model.predictions)

		# load local model attack graph
		if targeted_true:
			orig_img_loss = compute_cw_loss(target_model, orig_images, target_ys_one_hot,targeted=targeted_true)
		else:
			orig_img_loss = compute_cw_loss(target_model, orig_images,orig_labels,targeted=targeted_true)

		attack_sub_pgd_tar = LinfPGDAttack(local_model_ls,
							epsilon = li_eps, 
							k = k,
							a = a,
							random_start = False,
							loss_func = 'xent',
							targeted = targeted_true,
							x = x,
							y = y)
		# pgd attack to local models and generate adversarial example seed
		if targeted_true:
			_, pred_labs, local_aes,pgd_cnt_mat = local_attack_in_batches(sess, orig_images, target_ys_one_hot,eval_batch_size = 1,\
		attack_graph = attack_sub_pgd_tar,model = target_model,clip_min=clip_min,clip_max=clip_max)
		else:
			_, pred_labs, local_aes, pgd_cnt_mat  = local_attack_in_batches(sess, orig_images,orig_labels,eval_batch_size = 1,\
		attack_graph = attack_sub_pgd_tar,model = target_model,clip_min=clip_min,clip_max=clip_max)

		# calculate the loss for all adversarial seeds
		if targeted_true:
			adv_img_loss = compute_cw_loss(target_model,local_aes, target_ys_one_hot,targeted=targeted_true)
		else:
			adv_img_loss = compute_cw_loss(target_model,local_aes,orig_labels,targeted=targeted_true)

		
		
		success_rate = accuracy_score(target_ys, pred_labs)
		print('** Success rate of targeted adversarial examples generated from local models: **' + str(success_rate)) # keep
		accuracy = accuracy_score(np.argmax(orig_labels,axis = 1), pred_labs)
		print('** Success rate of targeted adversarial examples generated by local models (untargeted): **' + str(1-accuracy))

		# save local adversarial
		os.makedirs(prefix)
		# save statistics
		fname = prefix + '/adv_img_loss.txt'
		np.save(fname, adv_img_loss)
		fname = prefix + '/orig_img_loss.txt'
		np.save(fname, orig_img_loss)
		fname = prefix + '/pgd_cnt_mat.txt'
		np.save(fname, pgd_cnt_mat)

		# save output for local attacks
		fname = prefix + '/local_aes.npy'
		np.save(fname, local_aes)
		fname = prefix + '/orig_images.npy'
		np.save(fname, orig_images)
		fname = prefix + '/target_ys.npy'
		np.save(fname, target_ys)
		fname = prefix + '/target_ys_one_hot.npy'
		np.save(fname, target_ys_one_hot)
	else:
		print('loading data from files')
		local_aes = np.load(prefix + '/local_aes.npy')
		orig_images = np.load(prefix + '/orig_images.npy')
		target_ys = np.load(prefix + '/target_ys.npy')
		target_ys_one_hot = np.load(prefix + '/target_ys_one_hot.npy')
	assert local_aes.shape == (nb_imgs, 224, 224, 3)
	assert orig_images.shape == (nb_imgs, 224, 224, 3)
	assert target_ys.shape == (nb_imgs,)
	assert target_ys_one_hot.shape == (nb_imgs, class_num)

	# load autoencoder
	encoder = load_model(os.path.join(args["codec_dir"], 'imagenet_2_whole_encoder.h5'))
	decoder = load_model(os.path.join(args["codec_dir"], 'imagenet_2_whole_decoder.h5'))
	args["img_resize"] = decoder.input_shape[1]

	# ################## test whether autoencoder is working ##################
	# encode_img = encoder.predict(orig_images/255.-0.5)
	# decode_img = decoder.predict(encode_img)
	# # rescale decode_img
	# decode_img = np.clip((decode_img+0.5) * 255, a_min=0.0, a_max=255)
	# diff_img = (decode_img - orig_images) / 255.0 - 0.5
	# diff_mse = np.mean(diff_img.reshape(-1)**2)
	# print('MSE: %.4f' % diff_mse)
	########################################################################

	# define black-box model graph of autozoom
	blackbox_attack = AutoZOOM(sess, target_model, args, decoder,
							num_channels=3,image_size=224,num_labels=class_num)

	print('begin autoencoder attack')
	num_queries_list = []
	success_flags = []
	# fetch batch
	orig_images = orig_images[args['bstart']:args['bend']]
	target_ys = target_ys[args['bstart']:args['bend']]
	local_aes = local_aes[args['bstart']:args['bend']]
	target_ys_one_hot = target_ys_one_hot[args['bstart']:args['bend']]
	
	for idx in range(len(orig_images)):
		initial_img = orig_images[idx:idx+1]
		target_y_one_hot = target_ys_one_hot[idx]
		if args["attack_seed_type"] == 'adv':
			print('attack seed is %s'%args["attack_seed_type"])
			attack_seed = local_aes[idx:idx+1]
		else:
			print('attack seed is %s'%args["attack_seed_type"])
			attack_seed = orig_images[idx:idx+1]
		# scale imgs to [-0.5, 0.5]
		initial_img = initial_img / 255.0 - 0.5
		attack_seed = attack_seed / 255.0 - 0.5
		# attack
		if targeted_true:
			ae, query_num = autozoom_attack(blackbox_attack, attack_seed, initial_img, target_y_one_hot)
		else:
			raise NotImplementedError
		print('image %d: query_num %d'%(idx, query_num))
		# save query number and success
		if query_num == args["max_iterations"] * 2:
			success_flags.append(0)
		else:
			success_flags.append(1)
		num_queries_list.append(query_num)
	# save query number and success
	fname = prefix + '/{}_num_queries.txt'.format(args["attack_seed_type"])
	np.savetxt(fname,num_queries_list)
	fname = prefix + '/{}_success_flags.txt'.format(args["attack_seed_type"])
	np.savetxt(fname,success_flags)

	print('finish autozoom attack')

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("-b", "--batch_size", type=int, default=1, help="the batch size for zoo, zoo_ae attack")
	parser.add_argument("-c", "--init_const", type=float, default=1, help="the initial setting of the constant lambda")
	parser.add_argument("-d", "--dataset", default="imagenet", choices=["mnist", "cifar10", "cifar10_simple","imagenet", "imagenet_np"])
	parser.add_argument("-n", "--num_img", type=int, default=100, help = "number of test images to attack")
	parser.add_argument("-m", "--max_iterations", type=int, default=50000, help = "set 0 to use default value")
	parser.add_argument("-p", "--print_every", type=int, default=1000, help="print information every PRINT_EVERY iterations")
	parser.add_argument("-s", "--save_path", default=None, help="the path to save the results")
	parser.add_argument("--attack_type", default="targeted", choices=["targeted", "untargeted"], help="the type of attack")
	parser.add_argument("--confidence", default=0, type=float, help="the attack confidence")
	parser.add_argument("--codec_prefix", default=None, help="the coedec prefix, load the default codec is not set")
	parser.add_argument("--random_target", action="store_true", help="if set, choose random target, otherwise attack every possible target class, only works when ATTACK_TYPE=targeted")
	parser.add_argument("--num_rand_vec", type=int, default=1, help="the number of random vector for post success iteration")
	parser.add_argument("--seed", type=int, default=1234, help="random seed")
	parser.add_argument("--img_offset", type=int, default=0, help="the offset of the image index when getting attack data")
	parser.add_argument("--img_resize", default=None, type=int, help = "this option only works for ATTACK METHOD zoo and zoo_rv")
	parser.add_argument("--switch_iterations", type=int, default=1000, help="the iteration number for dynamic switching")
	parser.add_argument("--imagenet_dir", default=None, help="the path for imagenet images")
	parser.add_argument("--attack_single_img", default=None, help="attack a specific image, only works when DATASET is imagenet")
	parser.add_argument("--single_img_target_label", type=int, default=None, help="the target label for the single image attack")
	parser.add_argument("--compress_mode", type=int, default=None, help="specify the compress mode if autoencoder is used")
	parser.add_argument("--train_batch_size",type = int, default = 128, help = "size of training batches")
	parser.add_argument("--learning_rate",type = int, default = 0.002, help = 'learning rate for model training')  
	parser.add_argument("--nb_epochs", type =int, default = 10, help = "number of epcosh to train models")
	parser.add_argument("--codec_dir", type =str, default='./codec/', help = "autoencoder data path")
	#parameters tailored for batch attacks
	parser.add_argument("-c_val", "--cost_threshold", type=float, default=12.0/255, help="Feature modification cost (sqrt)")
	parser.add_argument("-z", "--use_zvalue", action='store_true') # just to help load the model trained on ZOO.
	parser.add_argument("--dist_metric", default="li", choices=["l2", "li"], help="norm ball for attacks")
	parser.add_argument("--no_save_img", action='store_true')# no need to save images
	parser.add_argument("--no_save_text", action='store_true')# no need to save texts
	parser.add_argument("-mt","--model_type", default="keras", choices=["keras", "robust"], help="the type of model used for attack")
	parser.add_argument("--file_path", default="sub_saved/cifar10-model", help="the type of model used for attack")
	parser.add_argument("--with_local",action = "store_true",help = "Conduct hybrid attacks")
	parser.add_argument("--load_existing",action = "store_true",help = "start from an already well trained models")
	parser.add_argument("--load_imgs",action = "store_true",help = "directly load saved data as attack candidate seeds")
	parser.add_argument("--use_loc_adv_thres", type = float, default = 0.2, help = "transfer rate threshold where targeted attack is more reliable...")  
	parser.add_argument("--fine_tune_freq", type = int, default = 20, help = "frequency to check the transfer rates")  
	parser.add_argument("--nb_epochs_sub", type = int, default = 20, help = "number of epcosh to train models") 
	parser.add_argument("--load_sub_model",action = "store_true",help = "load the pretrained model without any interactions")
	parser.add_argument("--sort_metric",default="min", choices=["max", "min","random"], help="the instance selection criteria of zoo attacks (sorted order of loss function)")
	parser.add_argument("--by_class",action = "store_true",help = "generate instances from each classes")
	parser.add_argument("--simple_target_model",action = "store_true",help = "use simple target model instead of advanced models")
	parser.add_argument("--simple_local_model",action = "store_true",help = "use simple local model instead of advanced models")
	parser.add_argument("--no_save_model",action = "store_true",help = "if yes, local models will not be saved")
	parser.add_argument("--use_mixup",action = "store_true",help = "if yes, adopt the mixup paper data augmentation method")
	parser.add_argument("--robust_type", default="madry", choices=["madry", "percy", "zico"], help="robust models to be attacked, only used when attacking robust models")
	parser.add_argument("--attack_seed_type", default="adv", choices=["orig", "adv"], help="attack seed type for batch attack")
	parser.add_argument('--bstart', type=int, default=0)
	parser.add_argument('--bend', type=int, default=100)
	args = vars(parser.parse_args())

	# settings based on dataset and attack method
	# mnist
	if args["dataset"] == "mnist":
		if args["max_iterations"] == 0:
			args["max_iterations"] = 2000
		args["use_tanh"] = False
		if args["codec_prefix"] is None:
			args["codec_prefix"] = "codec/mnist_1"
		args["lr"] = 1e-2
		args["compress_mode"] = 1
	# cifar10
	if args["dataset"] == "cifar10":
		if args["max_iterations"] == 0:
			args["max_iterations"] = 2000

		args["use_tanh"] = True
		if args["codec_prefix"] is None:
			args["codec_prefix"] = "codec/cifar10_2"
		args["lr"] = 1e-2
		if args["compress_mode"] is None:
			args["compress_mode"] = 2
	# imagenet
	if args["dataset"] == "imagenet" or args["dataset"] == "imagenet_np":
		if args["max_iterations"] == 0:
			if args["attack_method"] == "zoo_rv" or args["attack_method"] == "autozoom":
				args["max_iterations"] = 50000
			else:
				 args["max_iterations"] = 50000

			if not args["random_target"] and args["attack_type"] == "targeted":
				print("WARNING: You are trying to attack imagenet data with all (1000) labels.")

		if args["attack_single_img"] is not None:
			print("Imagenet targeting on one file:{}".format(args["attack_single_img"]))
			# force test image num to be 1
			args["num_img"] = 1
			if args["attack_type"] == "targeted":
				if args["single_img_target_label"] is None:
					print("Attack target is not set.")
					args["single_img_target_label"] = np.random.choice(range(1, 1001), 1)
					print("Randomly choose target label:{}".format(args["single_img_target_label"]))
				else:
					print("Targeting label:{}".format(args["single_img_target_label"]))
		# else:
		#     if args["imagenet_dir"] is None:
		#         raise Exception("Selecting imagenet as dataset but the path to the imagenet images are not set.")

		args["use_tanh"] = True
		args["lr"] = 2e-3
		
		if args["codec_prefix"] is None:
			args["codec_prefix"] = "codec/imagenet_3"

		args["compress_mode"] = 2

	if args["img_resize"] is not None:
		if args["attack_method"] == "zoo_ae" or args["attack_method"] == "autozoom":
			print("Attack method {} cannot use option img_resize, arugment ignored".format(args["attack_method"]))

	if args["save_path"] is None:
		# use dataset and attack method for the saving path
		args["save_path"] = "Results"

	# setup random seed
	random.seed(args["seed"])
	np.random.seed(args["seed"])
	tf.set_random_seed(args["seed"])
	print(args)
	main(args)
