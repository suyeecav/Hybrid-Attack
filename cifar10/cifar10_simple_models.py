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
from setup_cifar import CIFAR
# from tensorflow.python.platform import flags
# FLAGS = flags.FLAGS
IMAGE_ROWS = 32
IMAGE_COLS = 32
NUM_CHANNELS  = 3
NUM_CLASSES = 10

from keras.optimizers import SGD
from keras.utils import np_utils

def set_mnist_flags():
	# try:
	#     flags.DEFINE_integer('BATCH_SIZE', 64, 'Size of training batches')
	# except argparse.ArgumentError:
	#     pass
	flags.DEFINE_integer('BATCH_SIZE', 128, 'Size of training batches')
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

# carlini model, use it as target model
def modelA(input_ph):
	model = Sequential()
	
	model.add(Conv2D(64, (3, 3),
							input_shape=(IMAGE_ROWS,
										 IMAGE_COLS,
										 NUM_CHANNELS)))
	model.add(Activation('relu'))
	model.add(Conv2D(64, (3, 3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Conv2D(128, (3, 3)))
	model.add(Activation('relu'))
	model.add(Conv2D(128, (3, 3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Flatten())
	model.add(Dense(256))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(256))
	model.add(Activation('relu'))
	model.add(Dense(NUM_CLASSES))
	logits_tensor = model(input_ph)
	model.add(Activation('softmax'))

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
	model.add(Conv2D(32, (3, 3), padding='same',
					input_shape=(IMAGE_ROWS,
								 IMAGE_COLS,
								 NUM_CHANNELS)))
	model.add(Activation('relu'))
	model.add(Conv2D(32, (3, 3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	model.add(Conv2D(64, (3, 3), padding='same'))
	model.add(Activation('relu'))
	model.add(Conv2D(64, (3, 3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	model.add(Flatten())
	model.add(Dense(512))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(NUM_CLASSES))
	logits_tensor = model(input_ph)
	model.add(Activation('softmax'))

	return model, logits_tensor

def modelE(input_ph):

	model = Sequential()
	model.add(Convolution2D(48, 3, 3, 
							border_mode='same', 
							input_shape=(IMAGE_ROWS,
										 IMAGE_COLS,
										 NUM_CHANNELS)))
	model.add(Activation('relu'))
	model.add(Convolution2D(48, 3, 3))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))
	model.add(Convolution2D(96, 3, 3, 
							border_mode='same'))
	model.add(Activation('relu'))
	model.add(Convolution2D(96, 3, 3))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))
	model.add(Convolution2D(192, 3, 3,
							border_mode='same'))
	model.add(Activation('relu'))
	model.add(Convolution2D(192, 3, 3))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))
	model.add(Flatten())
	model.add(Dense(512))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(256))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(NUM_CLASSES))
	logits_tensor = model(input_ph)
	model.add(Activation('softmax'))

	return model, logits_tensor


def model_cifar10(input_ph,type=1):
	"""
	Defines MNIST model using Keras sequential model
	"""

	models = [modelA, modelB, modelC, modelD, modelE]
	model = models[type]
	return model(input_ph)


def data_gen_mnist(X_train):
	datagen = ImageDataGenerator()

	datagen.fit(X_train)
	return datagen

# a unified framework for all mnist models
class cifar10_models_simple(object):
	def __init__(self, sess, test_batch_size, type = 1,use_softmax = True, x = None,y = None, is_training=None,\
		 keep_prob=None,load_existing = False, model_name = 'modelA', loss = 'cw'):
		self.x = x
		self.y = y
		self.sess = sess
		self.is_training = is_training
		self.keep_prob = keep_prob
		self.test_batch_size = test_batch_size
		if load_existing:
			save_dir = '' # TODO: put your own directory
			filepath = os.path.join(save_dir, model_name+'.h5')
			model = load_model(filepath)
			self.model = model
			model = KerasModelWrapper(model)
			self.predictions = model.get_logits(self.x)
		else:
			model, preds = model_cifar10(input_ph = x, type = type)
			self.model = model
			self.predictions = preds

		self.probs = tf.nn.softmax(logits = self.predictions)
		self.eval_preds = tf.argmax(self.predictions, 1)
		self.y_target = tf.placeholder(tf.int64, shape=None) # tensor.shape (?,)
		self.eval_percent_adv = tf.equal(self.eval_preds, self.y_target) # one-to-one comparison
		if loss == 'cw':
			self.target_logits = tf.reduce_sum(self.y*self.predictions, 1)
			self.other_logits = tf.reduce_max((1-self.y)*self.predictions - (self.y*10000), 1)
			self.loss = self.other_logits - self.target_logits
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


def main(path_dir,trainable,type,epochs):
	# load the data first
	x = tf.placeholder(tf.float32, shape=(None, 32, 32, 3)) 
	data_augmentation = True
	data = CIFAR()
	X_train, Y_train, X_test, Y_test = data.train_data, data.train_labels, data.test_data, data.test_labels
	model_names = ['modelA','modelB','modelC','modelD','modelE']
	model, logits_tensor = model_cifar10(input_ph = x, type = type)
	path_dir = '' # TODO: put your own directory
	# initiate RMSprop optimizer
	# opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
	opt = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
	# Let's train the model using RMSprop
	model.compile(loss='categorical_crossentropy',
				optimizer=opt,
				metrics=['accuracy'])
	batch_size = 128
	if trainable:
		if not data_augmentation:
			print('Not using data augmentation.')
			model.fit(X_train, Y_train,
					batch_size=batch_size,
					epochs=epochs,
					validation_data=(X_test, Y_test),
					shuffle=True)
		else:
			print('Using real-time data augmentation.')
			# This will do preprocessing and realtime data augmentation:
			datagen = ImageDataGenerator(
				featurewise_center=False,  # set input mean to 0 over the dataset
				samplewise_center=False,  # set each sample mean to 0
				featurewise_std_normalization=False,  # divide inputs by std of the dataset
				samplewise_std_normalization=False,  # divide each input by its std
				zca_whitening=False,  # apply ZCA whitening
				zca_epsilon=1e-06,  # epsilon for ZCA whitening
				rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
				# randomly shift images horizontally (fraction of total width)
				width_shift_range=0.1,
				# randomly shift images vertically (fraction of total height)
				height_shift_range=0.1,
				shear_range=0.,  # set range for random shear
				zoom_range=0.,  # set range for random zoom
				channel_shift_range=0.,  # set range for random channel shifts
				# set mode for filling points outside the input boundaries
				fill_mode='nearest',
				cval=0.,  # value used for fill_mode = "constant"
				horizontal_flip=True,  # randomly flip images
				vertical_flip=False,  # randomly flip images
				# set rescaling factor (applied before any other transformation)
				rescale=None,
				# set function that will be applied on each input
				preprocessing_function=None,
				# image data format, either "channels_first" or "channels_last"
				data_format=None,
				# fraction of images reserved for validation (strictly between 0 and 1)
				validation_split=0.0)

			# Compute quantities required for feature-wise normalization
			# (std, mean, and principal components if ZCA whitening is applied).
			datagen.fit(X_train)

			# Fit the model on the batches generated by datagen.flow().
			model.fit_generator(datagen.flow(X_train, Y_train,
											batch_size=batch_size),
								epochs=epochs,
								validation_data=(X_test, Y_test),
								workers=4)
		# save model and weights
		model_path = path_dir + model_names[type]+'.h5'
		model.save(model_path)
		print('Saved trained model at %s ' % model_path)
	else:
		model = load_model(path_dir + model_names[type]+'.h5')

	temp = np.argmax(model.predict(X_test),axis=1)
	accuracy = accuracy_score(np.argmax(Y_test,axis=1), temp)
	print("oracle model accuracy:",accuracy)
	score = model.evaluate(X_test, Y_test, verbose=0)

	# Score trained model.
	scores = model.evaluate(X_test, Y_test, verbose=1)
	print('Test loss:', scores[0])
	print('Test accuracy:', scores[1])

if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument("--path_dir", type=str, default = "model/cifar10/",help="path to model")
	parser.add_argument("--type", type=int, help="model type", default=1)
	parser.add_argument("--epochs", type=int, default=50, help="number of epochs")
	parser.add_argument("--trainable", type=int, default = 1, help="decide if to train or directly load models")


	args = parser.parse_args()
	main(args.path_dir, args.trainable,args.type,args.epochs)

	