import argparse
import os
import math
import numpy as np

import tensorflow as tf
import keras
from sklearn.metrics import accuracy_score
from keras.utils import np_utils

from utils import ImageNetProducer, DataSpec, compute_cw_loss, keras_model_wrapper, load_model
from attack_local_ensemble import LinfPGDAttack
from cleverhans.utils_tf import model_eval
from attack_utils import nes_attack

###### put imagenet evaluation dataset path here ######
INPUT_DIR = '/bigtemp/jc6ub/imagenet_tf/val/' # change here for your imagenet data path
TOT_IMAGES = 200 # total image load from dataset
IMAGE_SIZE = (224,224,3,)

# setup parameters for nes attack
parser = argparse.ArgumentParser()
parser.add_argument('--samples-per-draw', type=int, default=50)
parser.add_argument('--nes_batch-size', type=int, default=50)
parser.add_argument('--sigma', type=float, default=1e-3*255)
parser.add_argument('--epsilon', type=float, default=12.0)
parser.add_argument('--learning-rate', type=float, default=2.55)
parser.add_argument('--log-iters', type=int, default=1)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--max-queries', type=int, default=100000)
parser.add_argument('--save-iters', type=int, default=50)
parser.add_argument('--plateau-drop', type=float, default=2.0) 
parser.add_argument('--min-lr-ratio', type=int, default=200)
parser.add_argument('--plateau-length', type=int, default=5)
parser.add_argument('--max-lr', type=float, default=255*1e-2)
parser.add_argument('--min-lr', type=float, default=255*5e-5)
parser.add_argument('--attack_type',type=str, choices=['targeted', 'untargeted'],  default='targeted')
parser.add_argument('--upper', type=float, default=255.0)
parser.add_argument('--lower', type=float, default=0.0)
parser.add_argument('--bstart', type=int, default=0)
parser.add_argument('--bend', type=int, default=100)
parser.add_argument('--K', type=int, default=30)
# augment for batch attack
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--nb_imgs', type=int, default=100)
parser.add_argument('--attack_seed_type',type=str, choices=['orig', 'adv'],  default='adv')

args = parser.parse_args()

# evaluate the accuracy of models on batches
def local_attack_in_batches(sess, data, labels, eval_batch_size, attack_graph,model=None, clip_min = 0, clip_max = 1):
	# Iterate over the samples batch-by-batch
	num_eval_examples = len(data)
	num_batches = int(math.ceil(num_eval_examples / eval_batch_size))
	local_aes = [] # adv accumulator
	pgd_cnt_mat = []
	for ibatch in range(num_batches):
		bstart = ibatch * eval_batch_size
		bend = min(bstart + eval_batch_size, num_eval_examples)
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
	print(corr_cls_idx.shape)
	print(corr_cls_idx)
	orig_images = x_test[corr_cls_idx]
	orig_labels = y_test[corr_cls_idx]
	target_ys = np.argmin(y_probs, axis = 1)[corr_cls_idx]
	target_ys_one_hot = np_utils.to_categorical(target_ys, class_num)

	return target_ys_one_hot, orig_images, target_ys, orig_labels

def main():
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
	target_model = keras_model_wrapper(pretrained_model, preprocess_input, x = x, y = y)
	for (indices, label, names, images) in image_producer.batches(sess):
		images = np.array(images)
		label = np_utils.to_categorical(np.array(label), class_num)
	accuracy = model_eval(sess, x, y, target_model.predictions, images, label, args= {'batch_size': 32})
	print('Test accuracy of wrapped target model:{:.4f}'.format(accuracy))
	
	# data information
	x_test, y_test = images, label # x_test [0, 255]
	print('loading %s images in total ', images.shape)
	print(np.min(x_test), np.max(x_test))

	# local attack specific parameters 
	clip_min = args.lower
	clip_max = args.upper
	nb_imgs = args.nb_imgs
	li_eps = args.epsilon
	targeted_true = True if args.attack_type == 'targeted' else False
	k = args.K # iteration
	a = args.learning_rate # step size

	# Test the accuracy of targeted attacks, need to redefine the attack graph
	target_ys_one_hot, orig_images, target_ys, orig_labels = generate_attack_inputs(target_model, x_test, y_test, class_num, nb_imgs)

	# Set random seed to improve reproducibility
	tf.set_random_seed(args.seed)
	np.random.seed(args.seed)

	# test whether adversarial examples exsit, if no, generate it, otherwise, load it.
	prefix = "Results"
	prefix = os.path.join(prefix, str(args.seed))

	if not os.path.exists(prefix): # no history info
		# load local models or define the architecture
		local_model_types = ['VGG16', 'VGG19', 'resnet50']
		local_model_ls = []
		pred_ls = []
		for model_type in local_model_types:
			pretrained_model, preprocess_input_func = load_model(model_type)
			local_model = keras_model_wrapper(pretrained_model, preprocess_input_func, x = x,y = y)
			accuracy = model_eval(sess, x, y, local_model.predictions, images, label, args= {'batch_size': 32})
			print('Test accuracy of model {}: {:.4f}'.format(model_type, accuracy))
			local_model_ls.append(local_model)
			pred_ls.append(local_model.predictions)

		# load local model attack graph
		if targeted_true:
			orig_img_loss = compute_cw_loss(target_model, orig_images, target_ys_one_hot, targeted=targeted_true)
		else:
			orig_img_loss = compute_cw_loss(target_model, orig_images, orig_labels, targeted=targeted_true)

		local_attack_graph = LinfPGDAttack(local_model_ls,
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
			_, pred_labs, local_aes, pgd_cnt_mat = local_attack_in_batches(sess, orig_images, target_ys_one_hot, eval_batch_size = 1,\
		attack_graph=local_attack_graph, model=target_model, clip_min=clip_min, clip_max=clip_max)
		else:
			_, pred_labs, local_aes, pgd_cnt_mat = local_attack_in_batches(sess, orig_images, orig_labels,eval_batch_size = 1,\
		attack_graph=local_attack_graph, model=target_model, clip_min=clip_min, clip_max=clip_max)

		# calculate the loss for all adversarial seeds
		if targeted_true:
			adv_img_loss = compute_cw_loss(target_model, local_aes, target_ys_one_hot,targeted=targeted_true)
		else:
			adv_img_loss = compute_cw_loss(target_model, local_aes, orig_labels,targeted=targeted_true)

		success_rate = accuracy_score(target_ys, pred_labs)
		print('** Success rate of targeted adversarial examples generated from local models: **' + str(success_rate))
		accuracy = accuracy_score(np.argmax(orig_labels, axis = 1), pred_labs)
		print('** Success rate of targeted adversarial examples generated by local models (untargeted): **' + str(1-accuracy))

		# l-inf distance of orig_images and local_aes
		dist = local_aes - orig_images
		l_fin_dist = np.linalg.norm(dist.reshape(nb_imgs, -1), np.inf, axis=1)

		# save the generated local adversarial example ...
		os.makedirs(prefix)
		# save statistics
		fname = prefix + '/adv_img_loss.npy'
		np.save(fname, adv_img_loss)
		fname = prefix + '/orig_img_loss.npy'
		np.save(fname, orig_img_loss)
		fname = prefix + '/pgd_cnt_mat.npy'
		np.save(fname, pgd_cnt_mat)
		# save output for local attacks
		fname = os.path.join(prefix, 'local_aes.npy')
		np.save(fname, local_aes)
		fname = os.path.join(prefix, 'orig_images.npy')
		np.save(fname, orig_images)
		fname = os.path.join(prefix, 'target_ys.npy')
		np.save(fname, target_ys)
		fname = os.path.join(prefix, 'target_ys_one_hot.npy')
		np.save(fname, target_ys_one_hot)
	else:
		print('loading data from files')
		local_aes = np.load(os.path.join(prefix, 'local_aes.npy'))
		orig_images = np.load(os.path.join(prefix, 'orig_images.npy'))
		target_ys = np.load(os.path.join(prefix, 'target_ys.npy'))
		target_ys_one_hot = np.load(os.path.join(prefix, 'target_ys_one_hot.npy'))

	assert local_aes.shape == (nb_imgs, 224, 224, 3)
	assert orig_images.shape == (nb_imgs, 224, 224, 3)
	assert target_ys.shape == (nb_imgs,)
	assert target_ys_one_hot.shape == (nb_imgs, class_num)

	print('begin NES attack')
	num_queries_list = []
	success_flags = []
	# fetch batch
	orig_images = orig_images[args.bstart:args.bend]
	target_ys = target_ys[args.bstart:args.bend]
	local_aes = local_aes[args.bstart:args.bend]
	# begin loop
	for idx in range(len(orig_images)):
		initial_img = orig_images[idx:idx+1]
		target_class = target_ys[idx]
		if args.attack_seed_type == 'adv':
			print('attack seed is %s'%args.attack_seed_type)
			attack_seed = local_aes[idx]
		else:
			print('attack seed is %s'%args.attack_seed_type)
			attack_seed = orig_images[idx]
		_, num_queries, adv = nes_attack(sess, args, target_model, attack_seed, initial_img, target_class, class_num, IMAGE_SIZE)
		if num_queries == args.max_queries:
			success_flags.append(0)
		else:
			success_flags.append(1)
		num_queries_list.append(num_queries)

	# save query number and success
	fname = os.path.join(prefix, '{}_num_queries.txt'.format(args.attack_seed_type))
	np.savetxt(fname, num_queries_list)
	fname = os.path.join(prefix, '{}_success_flags.txt'.format(args.attack_seed_type))
	np.savetxt(fname, success_flags)

	print('finish NES attack')
	
if __name__ == '__main__':
	main()
