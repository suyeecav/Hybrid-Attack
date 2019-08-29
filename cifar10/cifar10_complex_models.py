import keras
from keras.layers import Dense, Conv2D, Convolution2D, BatchNormalization, Activation, concatenate
from keras.layers import AveragePooling2D, Input, Flatten, Dropout, MaxPooling2D, GlobalAveragePooling2D
from keras.models import Model, Sequential
from keras.regularizers import l2

# custom for loading models
from cleverhans.utils_keras import KerasModelWrapper
import tensorflow as tf
import math
from keras.models import load_model
import os
import numpy as np
from keras.utils import np_utils

opt = 'adam'
def resnet_layer(inputs,
				 num_filters=16,
				 kernel_size=3,
				 strides=1,
				 activation='relu',
				 batch_normalization=True,
				 conv_first=True):
	"""2D Convolution-Batch Normalization-Activation stack builder

	# Arguments
		inputs (tensor): input tensor from input image or previous layer
		num_filters (int): Conv2D number of filters
		kernel_size (int): Conv2D square kernel dimensions
		strides (int): Conv2D square stride dimensions
		activation (string): activation name
		batch_normalization (bool): whether to include batch normalization
		conv_first (bool): conv-bn-activation (True) or
			bn-activation-conv (False)

	# Returns
		x (tensor): tensor as input to the next layer
	"""
	conv = Conv2D(num_filters,
				  kernel_size=kernel_size,
				  strides=strides,
				  padding='same',
				  kernel_initializer='he_normal',
				  kernel_regularizer=l2(1e-4))

	x = inputs
	if conv_first:
		x = conv(x)
		if batch_normalization:
			x = BatchNormalization()(x)
		if activation is not None:
			x = Activation(activation)(x)
	else:
		if batch_normalization:
			x = BatchNormalization()(x)
		if activation is not None:
			x = Activation(activation)(x)
		x = conv(x)
	return x

def resnet_v1(input_shape, depth, num_classes=10):
	"""ResNet Version 1 Model builder [a]

	Stacks of 2 x (3 x 3) Conv2D-BN-ReLU
	Last ReLU is after the shortcut connection.
	At the beginning of each stage, the feature map size is halved (downsampled)
	by a convolutional layer with strides=2, while the number of filters is
	doubled. Within each stage, the layers have the same number filters and the
	same number of filters.
	Features maps sizes:
	stage 0: 32x32, 16
	stage 1: 16x16, 32
	stage 2:  8x8,  64
	The Number of parameters is approx the same as Table 6 of [a]:
	ResNet20 0.27M
	ResNet32 0.46M
	ResNet44 0.66M
	ResNet56 0.85M
	ResNet110 1.7M

	# Arguments
		input_shape (tensor): shape of input image tensor
		depth (int): number of core convolutional layers
		num_classes (int): number of classes (CIFAR10 has 10)

	# Returns
		model (Model): Keras model instance
	"""
	
	if (depth - 2) % 6 != 0:
		raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')



	# Start model definition.
	num_filters = 16
	num_res_blocks = int((depth - 2) / 6)

	inputs = Input(shape=input_shape)
	x = resnet_layer(inputs=inputs)
	# Instantiate the stack of residual units
	for stack in range(3):
		for res_block in range(num_res_blocks):
			strides = 1
			if stack > 0 and res_block == 0:  # first layer but not first stack
				strides = 2  # downsample
			y = resnet_layer(inputs=x,
							 num_filters=num_filters,
							 strides=strides)
			y = resnet_layer(inputs=y,
							 num_filters=num_filters,
							 activation=None)
			if stack > 0 and res_block == 0:  # first layer but not first stack
				# linear projection residual shortcut connection to match
				# changed dims
				x = resnet_layer(inputs=x,
								 num_filters=num_filters,
								 kernel_size=1,
								 strides=strides,
								 activation=None,
								 batch_normalization=False)
			x = keras.layers.add([x, y])
			x = Activation('relu')(x)
		num_filters *= 2

	# Add classifier on top.
	# v1 does not use BN after last shortcut connection-ReLU
	x = AveragePooling2D(pool_size=8)(x)
	y = Flatten()(x)
	outputs = Dense(num_classes,
					activation='softmax',
					kernel_initializer='he_normal')(y)

	# Instantiate model.
	model = Model(inputs=inputs, outputs=outputs)
	return model


def resnet_v2(input_shape, depth, num_classes=10):
	"""ResNet Version 2 Model builder [b]

	Stacks of (1 x 1)-(3 x 3)-(1 x 1) BN-ReLU-Conv2D or also known as
	bottleneck layer
	First shortcut connection per layer is 1 x 1 Conv2D.
	Second and onwards shortcut connection is identity.
	At the beginning of each stage, the feature map size is halved (downsampled)
	by a convolutional layer with strides=2, while the number of filter maps is
	doubled. Within each stage, the layers have the same number filters and the
	same filter map sizes.
	Features maps sizes:
	conv1  : 32x32,  16
	stage 0: 32x32,  64
	stage 1: 16x16, 128
	stage 2:  8x8,  256

	# Arguments
		input_shape (tensor): shape of input image tensor
		depth (int): number of core convolutional layers
		num_classes (int): number of classes (CIFAR10 has 10)

	# Returns
		model (Model): Keras model instance
	"""
	if (depth - 2) % 9 != 0:
		raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')
	# Start model definition.
	num_filters_in = 16
	num_res_blocks = int((depth - 2) / 9)

	inputs = Input(shape=input_shape)
	# v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
	x = resnet_layer(inputs=inputs,
					 num_filters=num_filters_in,
					 conv_first=True)

	# Instantiate the stack of residual units
	for stage in range(3):
		for res_block in range(num_res_blocks):
			activation = 'relu'
			batch_normalization = True
			strides = 1
			if stage == 0:
				num_filters_out = num_filters_in * 4
				if res_block == 0:  # first layer and first stage
					activation = None
					batch_normalization = False
			else:
				num_filters_out = num_filters_in * 2
				if res_block == 0:  # first layer but not first stage
					strides = 2    # downsample

			# bottleneck residual unit
			y = resnet_layer(inputs=x,
							 num_filters=num_filters_in,
							 kernel_size=1,
							 strides=strides,
							 activation=activation,
							 batch_normalization=batch_normalization,
							 conv_first=False)
			y = resnet_layer(inputs=y,
							 num_filters=num_filters_in,
							 conv_first=False)
			y = resnet_layer(inputs=y,
							 num_filters=num_filters_out,
							 kernel_size=1,
							 conv_first=False)
			if res_block == 0:
				# linear projection residual shortcut connection to match
				# changed dims
				x = resnet_layer(inputs=x,
								 num_filters=num_filters_out,
								 kernel_size=1,
								 strides=strides,
								 activation=None,
								 batch_normalization=False)
			x = keras.layers.add([x, y])

		num_filters_in = num_filters_out

	# Add classifier on top.
	# v2 has BN-ReLU before Pooling
	x = BatchNormalization()(x)
	x = Activation('relu')(x)
	x = AveragePooling2D(pool_size=8)(x)
	y = Flatten()(x)
	outputs = Dense(num_classes,
					activation='softmax',
					kernel_initializer='he_normal')(y)

	# Instantiate model.
	model = Model(inputs=inputs, outputs=outputs)
	return model

# Code taken from https://github.com/geifmany/cifar-vgg
def vgg16_model(input_shape, num_classes=10):
	weight_decay = 0.0005
	
	model = Sequential()

	model.add(Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal',
					 input_shape=input_shape,kernel_regularizer=l2(weight_decay)))
	model.add(Activation('relu'))
	model.add(BatchNormalization())
	model.add(Dropout(0.3))

	model.add(Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay)))
	model.add(Activation('relu'))
	model.add(BatchNormalization())

	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay)))
	model.add(Activation('relu'))
	model.add(BatchNormalization())
	model.add(Dropout(0.4))

	model.add(Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay)))
	model.add(Activation('relu'))
	model.add(BatchNormalization())

	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay)))
	model.add(Activation('relu'))
	model.add(BatchNormalization())
	model.add(Dropout(0.4))

	model.add(Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay)))
	model.add(Activation('relu'))
	model.add(BatchNormalization())
	model.add(Dropout(0.4))

	model.add(Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay)))
	model.add(Activation('relu'))
	model.add(BatchNormalization())

	model.add(MaxPooling2D(pool_size=(2, 2)))


	model.add(Conv2D(512, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay)))
	model.add(Activation('relu'))
	model.add(BatchNormalization())
	model.add(Dropout(0.4))

	model.add(Conv2D(512, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay)))
	model.add(Activation('relu'))
	model.add(BatchNormalization())
	model.add(Dropout(0.4))

	model.add(Conv2D(512, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay)))
	model.add(Activation('relu'))
	model.add(BatchNormalization())

	model.add(MaxPooling2D(pool_size=(2, 2)))


	model.add(Conv2D(512, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay)))
	model.add(Activation('relu'))
	model.add(BatchNormalization())
	model.add(Dropout(0.4))

	model.add(Conv2D(512, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay)))
	model.add(Activation('relu'))
	model.add(BatchNormalization())
	model.add(Dropout(0.4))

	model.add(Conv2D(512, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay)))
	model.add(Activation('relu'))
	model.add(BatchNormalization())

	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.5))

	model.add(Flatten())
	model.add(Dense(512, kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay)))
	model.add(Activation('relu'))
	model.add(BatchNormalization())

	model.add(Dropout(0.5))
	model.add(Dense(num_classes, activation='softmax',
					kernel_initializer='he_normal'))

	return model

def densenet(input_shape, classes_num=10):
	def bn_relu(x):
		x = BatchNormalization()(x)
		x = Activation('relu')(x)
		return x

	def bottleneck(x):
		channels = growth_rate * 4
		x = bn_relu(x)
		x = Conv2D(channels,kernel_size=(1,1),strides=(1,1),padding='same',kernel_initializer='he_normal',kernel_regularizer=l2(weight_decay),use_bias=False)(x)
		x = bn_relu(x)
		x = Conv2D(growth_rate,kernel_size=(3,3),strides=(1,1),padding='same',kernel_initializer='he_normal',kernel_regularizer=l2(weight_decay),use_bias=False)(x)
		return x

	def single(x):
		x = bn_relu(x)
		x = Conv2D(growth_rate,kernel_size=(3,3),strides=(1,1),padding='same',kernel_initializer='he_normal',kernel_regularizer=l2(weight_decay),use_bias=False)(x)
		return x

	def transition(x, inchannels):
		outchannels = int(inchannels * compression)
		x = bn_relu(x)
		x = Conv2D(outchannels,kernel_size=(1,1),strides=(1,1),padding='same',kernel_initializer='he_normal',kernel_regularizer=l2(weight_decay),use_bias=False)(x)
		x = AveragePooling2D((2,2), strides=(2, 2))(x)
		return x, outchannels

	def dense_block(x,blocks,nchannels):
		concat = x
		for i in range(blocks):
			x = bottleneck(concat)
			concat = concatenate([x,concat], axis=-1)
			nchannels += growth_rate
		return concat, nchannels

	def dense_layer(x):
		return Dense(classes_num,activation='softmax',kernel_initializer='he_normal',kernel_regularizer=l2(weight_decay))(x)

	# parameter setting
	compression = 0.5
	growth_rate = 12
	depth = 100
	weight_decay = 0.0005
	nblocks = (depth - 4) // 6 
	nchannels = growth_rate * 2
	# build model 
	inputs = Input(shape=input_shape)
	x = Conv2D(nchannels,kernel_size=(3,3),strides=(1,1),padding='same',kernel_initializer='he_normal',kernel_regularizer=l2(weight_decay),use_bias=False)(inputs)

	x, nchannels = dense_block(x,nblocks,nchannels)
	x, nchannels = transition(x,nchannels)
	x, nchannels = dense_block(x,nblocks,nchannels)
	x, nchannels = transition(x,nchannels)
	x, nchannels = dense_block(x,nblocks,nchannels)
	x, nchannels = transition(x,nchannels)
	x = bn_relu(x)
	x = GlobalAveragePooling2D()(x)
	x = dense_layer(x)
	model = Model(inputs=inputs, outputs=x)
	return model

# a unified framework for all mnist models
class cifar10_models(object):
	def __init__(self, sess, depth, test_batch_size, use_softmax = True, x = None,y = None, load_existing = False, model_name = 'vgg16',loss = 'cw'):
		# "depth", "version" are required for resnet
		self.x = x
		self.y = y
		self.sess = sess
		self.test_batch_size = test_batch_size
		input_shape = (32,32,3)

		if load_existing:
			save_dir = 'CIFAR10_models/Normal_deep_models/' # TODO: replace with your own ROOT directory for normal cifar10 models
			if model_name == 'resnet_v1':
				model_load_name = 'cifar10_'+'ResNet'+str(depth)+'v1_model'
			elif model_name == 'resnet_v2':
				model_load_name = 'cifar10_'+'ResNet'+str(depth)+'v2_model'
			else:
				model_load_name = 'cifar10_'+model_name+'_model'
			filepath = os.path.join(save_dir, model_load_name +'.h5')
			model = load_model(filepath)
		else:
			if model_name == 'vgg16':
				model = vgg16_model(input_shape=input_shape)
			elif model_name == 'densenet':
				model = densenet(input_shape = input_shape)
			elif model_name == 'resnet_v1':
				model = resnet_v1(input_shape=input_shape,depth=depth)
			elif model_name == 'resnet_v2':
				model = resnet_v2(input_shape=input_shape,depth=depth)
			else:
				print("please provide a valid model name!")
				sys.exit(0)
			model.compile(loss='categorical_crossentropy',
			  optimizer=opt,
			  metrics=['accuracy'])
		self.model = model
		model = KerasModelWrapper(model)
		self.predictions = model.get_logits(self.x)
		self.probs = tf.nn.softmax(logits = self.predictions)
		self.eval_preds = tf.argmax(self.predictions, 1)
		self.y_target = tf.placeholder(tf.int64, shape=None) # tensor.shape (?,)
		self.eval_percent_adv = tf.equal(self.eval_preds, self.y_target) # one-to-one comparison
		if loss == 'cw':
			# tlab = tf.one_hot(self.y, NUM_CLASSES, on_value=1.0, off_value=0.0, dtype=tf.float32)
			target_probs = tf.reduce_sum(self.y*self.probs, 1)
			other_probs = tf.reduce_max((1-self.y)*self.probs - (self.y*10000), 1)
			self.loss = tf.log(other_probs + 1e-30) - tf.log(target_probs + 1e-30)
		elif loss == 'xent':
			self.loss = tf.nn.softmax_cross_entropy_with_logits(labels = self.y,logits = self.predictions)
		else:
			raise NotImplementedError

	def calcu_acc(self,data,lab):
		batch_size = self.test_batch_size
		#numpy operation to get prediction value
		corr_preds = 0
		num_batches = int(math.ceil(len(data) / batch_size))
		for ibatch in range(num_batches):
			bstart = ibatch * batch_size
			bend = min(bstart + batch_size, len(data))
			data_batch = data[bstart:bend,:]
			lab_batch = lab[bstart:bend]
			# acc calculation
			preds = self.sess.run(self.predictions,feed_dict = {self.x:data_batch})
			corr_preds += np.sum(np.argmax(lab_batch,axis = 1) == np.argmax(preds,axis = 1))
		return corr_preds/len(data)

	def predict_prob(self,data):
		batch_size = self.test_batch_size
		probs = []
		num_batches = int(math.ceil(len(data) / batch_size))
		for ibatch in range(num_batches):
			bstart = ibatch * batch_size
			bend = min(bstart + batch_size, len(data))
			data_batch = data[bstart:bend,:]
			# acc calculation
			prob = self.sess.run(self.probs,feed_dict = {self.x:data_batch})
			probs.extend(prob)
		return np.array(probs) 
	def pred_class(self,data):
		preds = self.predict_prob(data)
		labels = np.argmax(preds,axis = 1)
		return labels
	def eval_adv(self, adv, target_class):
		feed_dict = {self.x: adv, self.y_target: target_class}
		padv = self.sess.run(self.eval_percent_adv, feed_dict=feed_dict)
		return padv

	def get_loss(self, data, labels,class_num = 10):
		if len(labels.shape) == 1:
			labels = np_utils.to_categorical(labels, class_num)
		feed_dict = {self.x: data, self.y: labels}
		loss_val = self.sess.run(self.loss, feed_dict = feed_dict)
		return loss_val 