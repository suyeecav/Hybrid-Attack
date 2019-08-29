import sys
import os
import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import scipy.misc
import math
import time

#load model gragh
from robust_model_utils.madry_model import Model
from keras.utils import np_utils
from robust_model_utils.test_densenet_acc import wrap_as_densenet_model
from robust_model_utils.cifar10_vgg_train import wrap_as_vgg_model
from robust_model_utils.madry_thin_model import Thin_Model
class Load_Model:
	def __init__(self,sess):
		self.sess = sess
		pass
	def predict_prob(self):
		pass
	def predict(self):
		pass
	def correct_prediction(self):
		pass

# a unified framework for all cifar10 robust local models in tensorflow
class cifar10_tf_robust_models(object):
	def __init__(self, sess, test_batch_size, x = None,y = None, is_training=None,\
		 keep_prob=None, load_existing = True, model_name = 'adv_densenet',bias = 0.5, scale = 255,loss = 'xent'):
		# "depth", "version" are required for resnet
		self.x = x
		self.y = y
		self.is_training = is_training
		self.keep_prob = keep_prob
		self.bias = bias
		self.scale = scale
		self.sess = sess
		self.test_batch_size = test_batch_size

		old_vars = set(tf.global_variables())
		if model_name == 'adv_vgg':
			print("Adversarially trained VGG network currently is not compatible with the bbox attack, please use the other two models!")
			sys.exit(1)
			model = wrap_as_vgg_model(self.x,self.y)
			new_vars = set(tf.global_variables())
			save_dir = 'CIFAR10_models/Robust_Deep_models/Robust_VGG_local_model' # TODO: add your own directory of robust vgg model		
		elif model_name == 'adv_densenet':
			depth = 40
			model = wrap_as_densenet_model(depth,self.x,self.y,self.is_training,self.keep_prob,label_count=10)
			new_vars = set(tf.global_variables())
			save_dir = 'CIFAR10_models/Robust_Deep_models/Robust_DenseNet_local_model' # TODO: add your own directory path of robust densenet model
		elif model_name == 'adv_resnet':
			model = Thin_Model('eval',self.x,self.y)
			new_vars = set(tf.global_variables())
			save_dir = 'CIFAR10_models/Robust_Deep_models/Robust_ResNet_local_model' # TODO: add your own directory path of robust resnet model
		else:
			print("please provide a valid model name!")
			sys.exit(0)

		if load_existing:
			saver = tf.train.Saver(var_list = new_vars - old_vars)
			model_file = tf.train.latest_checkpoint(save_dir)
			if model_file is None:
				print('No model found')
				sys.exit()
			saver.restore(sess, model_file)

		self.model = model
		self.predictions = model.pre_softmax
		self.probs = tf.nn.softmax(logits = self.predictions)
		self.eval_preds = tf.argmax(self.predictions, 1)
		self.y_target = tf.placeholder(tf.int64, shape=None) # tensor.shape (?,)
		self.eval_percent_adv = tf.equal(self.eval_preds, self.y_target) # one-to-one comparison
		if loss == 'cw':
			# tlab = tf.one_hot(self.y, NUM_CLASSES, on_value=1.0, off_value=0.0, dtype=tf.float32)
			# target_probs = tf.reduce_sum(self.y*self.probs, 1)
			# other_probs = tf.reduce_max((1-self.y)*self.probs - (self.y*10000), 1)
			# self.loss = tf.log(other_probs + 1e-30) - tf.log(target_probs + 1e-30)
			self.target_logits = tf.reduce_sum(self.y*self.predictions, 1)
			self.other_logits = tf.reduce_max((1-self.y)*self.predictions - (self.y*10000), 1)
			self.loss = self.other_logits - self.target_logits
		elif loss == 'xent':
			if model_name == "adv_resnet":
				self.loss = model.y_xent
			else:
				self.loss = model.xent
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
			feed_dict = {self.x:(data_batch+self.bias)*self.scale,
			self.is_training:False,
			self.keep_prob:1}
			preds = self.sess.run(self.predictions,feed_dict = feed_dict)
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
			feed_dict = {self.x:(data_batch+self.bias)*self.scale,
			self.is_training:False,
			self.keep_prob:1}
			prob = self.sess.run(self.probs,feed_dict = feed_dict)
			probs.extend(prob)
		return np.array(probs) 
	def pred_class(self,data):
		preds = self.predict_prob(data)
		labels = np.argmax(preds,axis = 1)
		return labels
	def eval_adv(self, adv, target_class):
		# target_class: shape: (?,)
		feed_dict = {self.x: (adv+self.bias)*self.scale, 
		self.y_target: target_class,
		self.is_training:False,
		self.keep_prob:1}
		padv = self.sess.run(self.eval_percent_adv, feed_dict=feed_dict)
		return padv

	def get_loss(self, data, labels,class_num = 10):
		if len(labels.shape) == 1:
			labels = np_utils.to_categorical(labels, class_num)
		feed_dict = {self.x: (data+self.bias)*self.scale,
		self.y: labels,
		self.is_training:False,
		self.keep_prob:1}
		loss_val = self.sess.run(self.loss, feed_dict = feed_dict)
		return loss_val 

class Load_Madry_Model(Load_Model):
	def __init__(self,sess,model_dir, bias = 0.0, scale = 1.0,loss = 'cw'): 
		super().__init__(sess)
		#model load and restore weights
		model = Model(mode = 'eval')
		saver = tf.train.Saver()
		checkpoint = tf.train.latest_checkpoint(model_dir)
		saver.restore(sess, checkpoint)
		self.num_channels = model.num_channels
		self.image_size = model.image_size
		self.num_labels = model.num_labels
		self.bias = bias
		self.scale = scale
		self.model = model

		self.eval_logits = self.model.pre_softmax
		self.eval_probs = self.model.softmax_pred
		self.eval_preds = tf.argmax(self.eval_logits, 1)
		self.correct_prediction_tf = self.model.correct_prediction
		self.y_target = tf.placeholder(tf.int64, shape=None) # tensor.shape (?,)
		self.eval_percent_adv = tf.equal(self.eval_preds, self.y_target) # one-to-one comparison
		if loss == 'cw':
			tlab = tf.one_hot(self.model.y_input, self.num_labels, on_value=1.0, off_value=0.0, dtype=tf.float32)
			target_probs = tf.reduce_sum(tlab*self.eval_probs, 1)
			other_probs = tf.reduce_max((1-tlab)*self.eval_probs - (tlab*10000), 1)
			self.loss = tf.log(other_probs + 1e-30) - tf.log(target_probs + 1e-30)
		elif loss == 'xent':
			self.loss = self.model.y_xent
		else:
			raise NotImplementedError
		
	def predict_prob(self,data, batch_size = 1):
		# :param: label_tmp: not one hot encoded
		num_batches = int(math.ceil(len(data) / batch_size))
		probs = []
		for ibatch in range(num_batches):
			bstart = ibatch * batch_size
			bend = min(bstart + batch_size, len(data))
			x_batch = data[bstart:bend, :]
			feed_dict = {self.model.x_input:(x_batch+self.bias) * self.scale}
			probs.extend(self.sess.run(self.model.softmax_pred,feed_dict = feed_dict))
		return np.array(probs)

	def pred_class(self,data,batch_size = 1):
		# :param: label_tmp: not one hot encoded
		preds = self.predict_prob(data,batch_size)
		labels = np.argmax(preds,axis = 1)
		return labels
	
	def correct_prediction(self,data,label,batch_size = 100):
		num_batches = int(math.ceil(len(data) / batch_size))
		correct_prediction = []
		for ibatch in range(num_batches):
			bstart = ibatch * batch_size
			bend = min(bstart + batch_size, len(data))
			x_batch = data[bstart:bend, :]
			y_batch = label[bstart:bend]
			feed_dict = {self.model.x_input:(x_batch+self.bias) * self.scale,self.model.y_input:y_batch}
			correct_prediction.extend(self.sess.run(self.correct_prediction_tf, feed_dict = feed_dict))
		return np.array(correct_prediction)

	def eval_adv(self, adv, target_class):
		feed_dict = {self.model.x_input:(adv+self.bias) * self.scale, self.y_target: target_class}
		padv = self.sess.run(self.eval_percent_adv, feed_dict=feed_dict)
		return padv

	def get_loss(self, data, labels, class_num = 10):
		feed_dict = {self.model.x_input:(data+self.bias) * self.scale, self.model.y_input: labels}
		loss_val = self.sess.run(self.loss, feed_dict = feed_dict)
		return loss_val 

