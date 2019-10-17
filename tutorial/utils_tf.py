"""Utility functions and classes for handling image datasets."""

import os.path as osp
import os
import numpy as np
import tensorflow as tf
from keras.utils import np_utils

from cleverhans.utils_keras import KerasModelWrapper

def load_local_model(model_name):
	if model_name == 'resnet50':
		from keras.applications.resnet50 import ResNet50
		from keras.applications.resnet50 import preprocess_input
		pretrained_model = ResNet50(weights='imagenet')
	elif model_name == 'VGG16':
		from keras.applications.vgg16 import VGG16
		from keras.applications.vgg16 import preprocess_input
		pretrained_model = VGG16(weights='imagenet')
	elif model_name == 'VGG19':
		from keras.applications.vgg19 import VGG19
		from keras.applications.vgg19 import preprocess_input
		pretrained_model = VGG19(weights='imagenet')
	else:
		raise NotImplementedError

	return pretrained_model, preprocess_input


# directly wraps based on the model
class keras_model_wrapper():
	def __init__(self, model, preprocess, x = None,y = None):
		# :param: x: placeholder for inputs
		# :param: y: placeholder for labels
		self.x = x
		self.y = y
		self.keras_model = model
		model_wrap = KerasModelWrapper(model)
		self.preprocess = preprocess
		self.processed_image = self.preprocess(self.x)
		self.predictions = model_wrap.get_logits(self.processed_image) # logit
		self.eval_preds = tf.argmax(self.predictions, 1)
		self.probs = tf.nn.softmax(logits = self.predictions)
		self.loss = tf.nn.softmax_cross_entropy_with_logits(labels = self.y,
															logits = self.predictions)

	def predict_prob(self, x):
		x_copy = np.copy(x)
		if len(x_copy.shape) == 3:
			x_copy = np.expand_dims(x_copy, axis=0)
		process_input = self.preprocess(x_copy) # inplace change
		y_probs = self.keras_model.predict(process_input)
		return y_probs

	def pred_class(self, sess, x):
		if len(x.shape) == 3:
			x = np.expand_dims(x, axis=0)
		feed_dict = {self.x: x}
		preds = sess.run(self.eval_preds, feed_dict)
		return preds[0]

	def get_loss(self, sess, data, labels, class_num):
		if len(labels.shape) == 1:
			labels = np_utils.to_categorical(labels, class_num)
		feed_dict = {self.x: data, self.y: labels}
		loss_val = sess.run(self.loss, feed_dict = feed_dict) 
		return loss_val

	def eval_adv(self, sess, adv, target_class):
		if len(adv.shape) == 3:
			adv = np.expand_dims(adv, axis=0)
		feed_dict = {self.x: adv}
		preds = sess.run(self.eval_preds, feed_dict)
		assert  type(preds[0]) == type(target_class[0]), (type(preds[0]),type(target_class[0]))
		if preds[0] == target_class[0]:
			return True
		else:
			return False

def compute_cw_loss(model, data, labs, targeted = True):
	# CW loss will be calculated in this func
	# :param: model: provide prediction scores
	# :param: x: image instance, one hot form
	# :param: y: specified instance
	output = (model.predict_prob(data)) #shape:(?,class_num)
	real = np.sum(labs*output,axis = 1)
	other = np.amax((1-labs)*output - (labs*10000),1)
	if targeted:
		loss = np.maximum(0,np.log(other + 1e-30) - np.log(real + 1e-30))
	else:
		loss = np.maximum(0,np.log(real + 1e-30) - np.log(other + 1e-30))
	return loss #shape (?,)

def auto_str(cls):
	def __str__(self):
		return '%s(%s)' % (
			type(self).__name__,
			', '.join('%s=%s' % item for item in vars(self).items())
		)
	cls.__str__ = __str__
	return cls


@auto_str
class DataSpec(object):
	'''Input data specifications for an ImageNet model.'''

	def __init__(self,
				 batch_size,
				 scale_size,
				 crop_size,
				 isotropic,
				 channels=3,
				 rescale=[0.0, 255.0],
				 mean= np.array([0, 0, 0]), 
				 bgr=False):
				 #np.array([103.939, 116.779, 123.68])

		# The recommended batch size for this model
		self.batch_size = batch_size
		# The image should be scaled to this size first during preprocessing
		self.scale_size = scale_size
		# Whether the model expects the rescaling to be isotropic
		self.isotropic = isotropic
		# A square crop of this dimension is expected by this model
		self.crop_size = crop_size
		# The number of channels in the input image expected by this model
		self.channels = channels
		# The mean to be subtracted from each image. By default, the per-channel ImageNet mean.
		# The values below are ordered BGR, as many Caffe models are trained in this order.
		# Some of the earlier models (like AlexNet) used a spatial three-channeled mean.
		# However, using just the per-channel mean values instead doesn't
		# affect things too much.
		self.mean = mean
		# Whether this model expects images to be in BGR order
		self.expects_bgr = bgr
		self.rescale = rescale


#
def process_image(img, scale, isotropic, crop, mean, rescale, need_rescale):
	"""Crops, scales, and normalizes the given image.
	scale : The image wil be first scaled to this size.
			If isotropic is true, the smaller side is rescaled to this,
			preserving the aspect ratio.
	crop  : After scaling, a central crop of this size is taken.
	mean  : Subtracted from the image
	rescale: Rescale pixel value to [x, y]
	"""
	if need_rescale:
		# Rescale
		if isotropic:
			img_shape = tf.to_float(tf.shape(img)[:2])
			min_length = tf.minimum(img_shape[0], img_shape[1])
			new_shape = tf.to_int32((scale / min_length) * img_shape)
		else:
			new_shape = tf.stack([scale, scale])

		img = tf.image.resize_images(img, (new_shape[0], new_shape[1]))
		# Center crop
		# Use the slice workaround until crop_to_bounding_box supports deferred tensor shapes
		# See: https://github.com/tensorflow/tensorflow/issues/521
		offset = (new_shape - crop) // 2
		offset = tf.to_int32(offset)
		img = tf.slice(img, begin=tf.stack(
			[offset[0], offset[1], 0]), size=tf.stack([crop, crop, -1]))
	else:
		img = tf.image.resize_images(img, crop, crop)
	# Mean subtraction
	img = tf.to_float(img)
	[l, r] = rescale
	img = img / 255.0 * (r - l) + l
	img = img - mean
	return img

class ImageProducer (object):
	"""
	Loads and processes batches of images in parallel.
	"""

	def __init__(
			self,
			image_paths,
			need_rescale,
			data_spec,
			num_concurrent=1,
			batch_size=None,
			labels=None):
		# The data specifications describe how to process the image
		self.data_spec = data_spec
		# A list of full image paths
		self.image_paths = image_paths
		# Need to rescale images
		self.need_rescale = need_rescale
		# An optional list of labels corresponding to each image path
		self.labels = labels
		# A boolean flag per image indicating whether its a JPEG or PNG
		self.extension_mask = self.create_extension_mask(self.image_paths)

		self.startover_flag = True

		# Load images and save as cache
		self.setup(batch_size=batch_size)

	def setup(self, batch_size):
		# Validate the batch size
		num_images = len(self.image_paths)
		self.batch_size = min(num_images, batch_size or self.data_spec.batch_size)
		if num_images % batch_size != 0:
			raise ValueError(
				'The total number of images ({}) must be divisible by the batch size ({}).'.format(
					num_images, batch_size))
		self.num_batches = int(num_images / batch_size)

		self.img_cache = {}

		for idx in range(num_images):
			is_jpeg = self.extension_mask[idx]
			image_path = self.image_paths[idx]
			# Load the image
			img = self.load_image(image_path, is_jpeg)
			# Process the image
			processed_img = process_image(img=img,
										  scale=self.data_spec.scale_size,
										  isotropic=self.data_spec.isotropic,
										  crop=self.data_spec.crop_size,
										  mean=self.data_spec.mean,
										  rescale=self.data_spec.rescale,
										  need_rescale=self.need_rescale)
			self.img_cache[idx] = processed_img

	def startover(self):
		self.startover_flag = True

	def get(self, batch_idx, session):
		'''
		Get a single batch of images along with their indices. If a set of labels were provided,
		the corresponding labels are returned instead of the indices.
		'''

		indices = [batch_idx * self.batch_size + idx for idx in range(self.batch_size)]
		images = [self.img_cache[idx].eval(session=session) for idx in indices]
		labels = [self.labels[idx] for idx in indices]
		names = [osp.basename(osp.normpath(self.image_paths[idx]))
				 for idx in indices]
		return (indices, labels, names, images)

	def batches(self, session):
		'''Yield a batch until no more images are left.'''
		if self.startover_flag:
			for batch_idx in range(self.num_batches):
				yield self.get(batch_idx, session)
			self.start_over_flag = False

	def load_image(self, image_path, is_jpeg):
		# Read the file
		file_data = tf.read_file(image_path)
		# Decode the image data
		img = tf.cond(
			tf.logical_and(is_jpeg, True), lambda: tf.image.decode_jpeg(file_data, channels=self.data_spec.channels), 
					 lambda: tf.image.decode_png(file_data, channels=self.data_spec.channels))
		if self.data_spec.expects_bgr:
			# Convert from RGB channel ordering to BGR
			# This matches, for instance, how OpenCV orders the channels.
			img = tf.reverse(img, [-1])
		return img

	@staticmethod
	def create_extension_mask(paths):

		def is_jpeg(path):
			extension = osp.splitext(path)[-1].lower()
			if extension in ('.jpg', '.jpeg'):
				return True
			if extension != '.png':
				raise ValueError(
					'Unsupported image format: {}'.format(extension))
			return False 
		
		return [is_jpeg(p) for p in paths]

	@staticmethod
	def is_image(image_name):
		extension = osp.splitext(image_name)[-1].lower()
		if extension in ('.jpg', '.jpeg', '.png'):
			return True
		return False

	def __len__(self):
		return len(self.image_paths)


class ImageNetProducer(ImageProducer):

	@staticmethod
	def get_truth_labels(file_list):
		val_file_path = '/bigtemp/jc6ub/imagenet_tf/val.txt'
		label_finder = {}
		with open(val_file_path) as val_file:
			for line in val_file:
				(key, val) = line.split()
				label_finder[key[:23]] = int(val)

		def get_truth_label(file_name):
			file_name = file_name[:23]
			if file_name in label_finder:
				return label_finder[file_name]
			else:
				return -1

		return [get_truth_label(image_file_name)
				for image_file_name in file_list]

	@staticmethod
	def get_human_label(label_id):
		human_file_path = 'data/ilsvrc12/imagenet-classes.txt'
		descriptions = [line.strip() for line in open(human_file_path)]
		return descriptions[label_id]

	def __init__(
			self,
			data_path,
			num_images,
			data_spec,
			need_rescale=True,
			batch_size=None):
		# Read in the ground truth labels for the validation set
		# The get_ilsvrc_aux.sh in Caffe's data/ilsvrc12 folder can fetch a
		# copy of val.txt
		file_list = [image_name for image_name in os.listdir(
						data_path) if ImageNetProducer.is_image(image_name)]

		if len(file_list) > num_images:
			file_list = file_list[:num_images]

		image_paths = [osp.join(data_path, p) for p in file_list]
		# The corresponding ground truth labels
		labels = ImageNetProducer.get_truth_labels(file_list)
		# Initialize base
		super(ImageNetProducer, self).__init__(
			image_paths=image_paths,
			need_rescale=need_rescale,
			data_spec=data_spec,
			labels=labels,
			batch_size=batch_size)