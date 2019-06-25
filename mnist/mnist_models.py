from keras.datasets import mnist
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Convolution2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.models import load_model
import keras
from cleverhans.utils_keras import KerasModelWrapper
import tensorflow as tf

from distutils.version import LooseVersion
if LooseVersion(keras.__version__) >= LooseVersion('2.0.0'):
	from keras.layers import Conv2D
else:
	from keras.layers import Convolution2D

# import argparse
import numpy as np
from sklearn.metrics import accuracy_score
import math
import os
from keras.utils import np_utils

IMAGE_ROWS = 28
IMAGE_COLS = 28
NUM_CHANNELS  = 1
NUM_CLASSES = 10

def set_mnist_flags():
	# try:
	#     flags.DEFINE_integer('BATCH_SIZE', 64, 'Size of training batches')
	# except argparse.ArgumentError:
	#     pass
	flags.DEFINE_integer('BATCH_SIZE', 64, 'Size of training batches')
	flags.DEFINE_integer('NUM_CLASSES', 10, 'Number of classification classes')
	flags.DEFINE_integer('IMAGE_ROWS', 28, 'Input row dimension')
	flags.DEFINE_integer('IMAGE_COLS', 28, 'Input column dimension')
	flags.DEFINE_integer('NUM_CHANNELS', 1, 'Input depth dimension')


def data_mnist(one_hot=True):
	"""
	Preprocess MNIST dataset
	"""
	# the data, shuffled and split between train and test sets
	(X_train, y_train), (X_test, y_test) = mnist.load_data()

	y_train = y_train


	X_train = X_train.reshape(X_train.shape[0],
							  IMAGE_ROWS,
							  IMAGE_COLS,
							  NUM_CHANNELS)

	X_test = X_test.reshape(X_test.shape[0],
							IMAGE_ROWS,
							IMAGE_COLS,
							NUM_CHANNELS)

	X_train = X_train.astype('float32')
	X_test = X_test.astype('float32')
	X_train /= 255 - 0.5
	X_test /= 255 -0.5
	print('X_train shape:', X_train.shape)
	print(X_train.shape[0], 'train samples')
	print(X_test.shape[0], 'test samples')

	print("Loaded MNIST test data.")

	if one_hot:
		# convert class vectors to binary class matrices
		y_train = np_utils.to_categorical(y_train, NUM_CLASSES).astype(np.float32)
		y_test = np_utils.to_categorical(y_test, NUM_CLASSES).astype(np.float32)

	return X_train, y_train, X_test, y_test


def modelA(input_ph):
	model = Sequential()
	model.add(Conv2D(64, (5, 5),
							padding='valid'))
	model.add(Activation('relu'))

	model.add(Conv2D(64, (5, 5)))
	model.add(Activation('relu'))

	model.add(Dropout(0.25))

	model.add(Flatten())
	model.add(Dense(128))
	model.add(Activation('relu'))

	model.add(Dropout(0.5))
	model.add(Dense(NUM_CLASSES))
	logits_tensor = model(input_ph)
	model.add(Activation('softmax'))  #14

	return model, logits_tensor


def modelB(input_ph):
	model = Sequential()
	model.add(Dropout(0.2, input_shape=(IMAGE_ROWS,
										IMAGE_COLS,
										NUM_CHANNELS)))
	model.add(Convolution2D(64, (8, 8),
							subsample=(2, 2),
							border_mode='same'))
	model.add(Activation('relu'))

	model.add(Convolution2D(128, (6, 6),
							subsample=(2, 2),
							border_mode='valid'))
	model.add(Activation('relu'))

	model.add(Convolution2D(128, (5, 5),
							subsample=(1, 1)))
	model.add(Activation('relu'))

	model.add(Dropout(0.5))

	model.add(Flatten())
	model.add(Dense(NUM_CLASSES))
	logits_tensor = model(input_ph)
	model.add(Activation('softmax'))  #14

	return model, logits_tensor



def modelC(input_ph):
	model = Sequential()
	model.add(Convolution2D(128, (3, 3),
							border_mode='valid',
							input_shape=(IMAGE_ROWS,
										 IMAGE_COLS,
										 NUM_CHANNELS)))
	model.add(Activation('relu'))

	model.add(Convolution2D(64, (3, 3)))
	model.add(Activation('relu'))

	model.add(Dropout(0.25))

	model.add(Flatten())
	model.add(Dense(128))
	model.add(Activation('relu'))

	model.add(Dropout(0.5))
	model.add(Dense(NUM_CLASSES))
	logits_tensor = model(input_ph)
	model.add(Activation('softmax'))  #14

	return model, logits_tensor


def modelD(input_ph):
	model = Sequential()

	model.add(Flatten(input_shape=(IMAGE_ROWS,
								   IMAGE_COLS,
								   NUM_CHANNELS)))

	model.add(Dense(300, init='he_normal', activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(300, init='he_normal', activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(300, init='he_normal', activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(300, init='he_normal', activation='relu'))
	model.add(Dropout(0.5))

	model.add(Dense(NUM_CLASSES))
	logits_tensor = model(input_ph)
	model.add(Activation('softmax'))  #14

	return model, logits_tensor


def modelE(input_ph):
	model = Sequential()

	model.add(Flatten(input_shape=(IMAGE_ROWS,
								   IMAGE_COLS,
								   NUM_CHANNELS)))

	model.add(Dense(100, activation='relu'))
	model.add(Dense(100, activation='relu'))

	model.add(Dense(NUM_CLASSES))
	logits_tensor = model(input_ph)
	model.add(Activation('softmax'))  #14
	return model, logits_tensor

def modelF(input_ph):
	model = Sequential()

	model.add(Convolution2D(32, (3, 3),
							border_mode='valid',
							input_shape=(IMAGE_ROWS,
										 IMAGE_COLS,
										 NUM_CHANNELS)))
	model.add(Activation('relu'))

	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Convolution2D(64, (3, 3)))
	model.add(Activation('relu'))

	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Flatten())
	model.add(Dense(1024))
	model.add(Activation('relu'))

	model.add(Dense(NUM_CLASSES))
	logits_tensor = model(input_ph)
	model.add(Activation('softmax'))  #14

	return model, logits_tensor

def model_mnist(input_ph,type=1):
	"""
	Defines MNIST model using Keras sequential model
	"""

	models = [modelA, modelB, modelC, modelD, modelE, modelF]
	model = models[type]
	return model(input_ph)

def cnn_model(input_ph):
	input_shape = (28, 28, 1)
	model = Sequential()
	model.add(Conv2D(32, kernel_size=(3, 3),
					activation='relu',
					input_shape=input_shape))
	model.add(Conv2D(64, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(NUM_CLASSES))

	logits_tensor = model(input_ph)
	model.add(Activation('softmax'))
	return model,logits_tensor

def data_gen_mnist(X_train):
	datagen = ImageDataGenerator()

	datagen.fit(X_train)
	return datagen

# a unified framework for all mnist models
class mnist_models(object):
	def __init__(self, sess, type = 1,use_softmax = True, x = None,y = None, load_existing = False, model_name = 'modelA',loss = 'cw'):
		self.x = x
		self.y = y
		self.sess = sess
		if load_existing:
			save_dir = '' # TODO: put your own directory here
			filepath = os.path.join(save_dir, model_name+'.h5')
			model = load_model(filepath)
			self.model = model
			model = KerasModelWrapper(model)
			self.predictions = model.get_logits(self.x)
		else:
			model, preds = model_mnist(input_ph = x, type = type)
			self.model = model
			self.predictions = preds

		self.probs = tf.nn.softmax(logits = self.predictions)
		self.loss = tf.nn.softmax_cross_entropy_with_logits(labels = self.y,
			logits = self.predictions)

		if loss == 'cw':
			# tlab = tf.one_hot(self.y, NUM_CLASSES, on_value=1.0, off_value=0.0, dtype=tf.float32)
			target_probs = tf.reduce_sum(self.y*self.probs, 1)
			other_probs = tf.reduce_max((1-self.y)*self.probs - (self.y*10000), 1)
			self.loss = tf.log(other_probs + 1e-30) - tf.log(target_probs + 1e-30)
		elif loss == 'xent':
			self.loss = tf.nn.softmax_cross_entropy_with_logits(labels = self.y,logits = self.predictions)
		else:
			raise NotImplementedError
		self.eval_preds = tf.argmax(self.predictions, 1)
		self.y_target = tf.placeholder(tf.int64, shape=None) # tensor.shape (?,)
		self.eval_percent_adv = tf.equal(self.eval_preds, self.y_target) # one-to-one comparison

	def calcu_acc(self,data,lab,batch_size = 1):
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

	def predict_prob(self,data,batch_size = 1):
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
	def pred_class(self,data,batch_size = 1):
		preds = self.predict_prob(data,batch_size)
		labels = np.argmax(preds,axis = 1)
		return labels
	def get_loss(self, data, labels,class_num = 10):
		if len(labels.shape) == 1:
			labels = np_utils.to_categorical(labels, class_num)
		feed_dict = {self.x: data, self.y: labels}
		loss_val = self.sess.run(self.loss, feed_dict = feed_dict) 
		return loss_val
	def eval_adv(self, adv, target_class):
		feed_dict = {self.x: adv, self.y_target: target_class}
		padv = self.sess.run(self.eval_percent_adv, feed_dict=feed_dict)
		return padv

	