import math
import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F

from tensorflow.examples.tutorials.mnist import input_data
from robust_model_utils.madry_model import Model as madry
from robust_model_utils.zico_model import mnist_model as zico
from robust_model_utils.sdp_robust_model.percy_model import Model as percy
from keras.utils import np_utils

percy_weights_name = "sdp-weights" 
class PercyModel(object):
	def __init__(self, sess, num_labels=10, model_dir="models/percy/sdp-weights", bias = 0.0, loss='cw'):
		self.model = percy()
		self.bias = bias
		# self.sess = tf.InteractiveSession()
		self.sess = sess
		saver = tf.train.Saver()
		saver.restore(self.sess, model_dir + percy_weights_name)
		self.x = self.model.x_input # tensor.shape (?, 784)
		self.y = self.model.y_input # tensor.shape (?,)
		self.eval_logits = self.model.pre_softmax
		self.eval_probs = tf.nn.softmax(self.eval_logits)
		self.eval_preds = tf.argmax(self.eval_logits, 1)
		self.correct_prediction = self.model.correct_prediction
		self.y_target = tf.placeholder(tf.int64, shape=None) # tensor.shape (?,)
		self.eval_percent_adv = tf.equal(self.eval_preds[0], self.y_target) # one-to-one comparison
		if loss == 'cw':
			tlab = tf.one_hot(self.y, num_labels, on_value=1.0, off_value=0.0, dtype=tf.float32)
			target_probs = tf.reduce_sum(tlab*self.eval_probs, 1)
			other_probs = tf.reduce_max((1-tlab)*self.eval_probs - (tlab*10000), 1)
			self.loss = tf.log(other_probs + 1e-30) - tf.log(target_probs + 1e-30)
		elif loss == 'xent':
			self.loss = self.model.y_xent
		else:
			raise NotImplementedError

	# def __del__(self):
	#     self.sess.close()
	#     print('session close')

	def pred_class(self, data,batch_size = 1):
		if len(data.shape) > 2:
			data = data.reshape((-1,784)) 
		feed_dict = {self.x: data+self.bias}
		preds = self.sess.run(self.eval_preds, feed_dict)
		return preds

	def predict_prob(self, data,batch_size = 1):
		if len(data.shape) > 2:
			data = data.reshape((-1,784)) 
		feed_dict = {self.x: data+self.bias}
		probs = self.sess.run(self.eval_probs, feed_dict = feed_dict) 
		return probs # shape=(?,10)

	def get_loss(self, data, labels,class_num = 10):
		if len(data.shape) > 2:
			data = data.reshape((-1,784)) 
		feed_dict = {self.x: data+self.bias, self.y: labels}
		loss_val = self.sess.run(self.loss, feed_dict = feed_dict) 
		return loss_val

	def get_corr_pred_idx(self, all_images_o, all_labels,eval_batch_size = 100):
		"""find the correctly classfied image index of given data and model in MNIST data"""
		if len(all_images_o.shape) > 2:
			all_images = np.copy(all_images_o)
			all_images = all_images.reshape((-1,784)) 
		num_eval_examples = len(all_images)
		num_batches = int(math.ceil(num_eval_examples / eval_batch_size))
		corr_pred_val = []
		for ibatch in range(num_batches):
			bstart = ibatch * eval_batch_size
			bend = min(bstart + eval_batch_size, num_eval_examples)
			x_batch = all_images[bstart:bend, :]
			y_batch = all_labels[bstart:bend]
			feed_dict= {self.x: x_batch+self.bias, self.y: y_batch}
			corr_pred_val.append(self.sess.run(self.correct_prediction, feed_dict = feed_dict))
		corr_pred_val = np.concatenate(corr_pred_val, axis=0)
		corr_pred_idx = np.nonzero(corr_pred_val)
		return corr_pred_idx[0]
	
	def eval_adv(self, adv, target_class):
		feed_dict = {self.x: adv+self.bias, self.y_target: target_class}
		padv = self.sess.run(self.eval_percent_adv, feed_dict=feed_dict)
		return padv

class MadryModel(object):
	def __init__(self, sess,num_labels=10, model_dir="models/adv_trained_mnist", bias = 0.0, loss='cw'):
		#model load and restore weights
		self.model = madry()
		self.bias = bias
		self.sess = sess
		saver = tf.train.Saver()
		checkpoint = tf.train.latest_checkpoint(model_dir)
		saver.restore(self.sess, checkpoint)
		# connect the model graph interface
		self.x = self.model.x_input # tensor.shape (?, 784)
		self.y = self.model.y_input # tensor.shape (?,)
		self.eval_logits = self.model.pre_softmax
		self.eval_probs = tf.nn.softmax(self.eval_logits)
		self.eval_preds = tf.argmax(self.eval_logits, 1)
		self.correct_prediction = self.model.correct_prediction
		self.y_target = tf.placeholder(tf.int64, shape=None) # tensor.shape (?,)
		self.eval_percent_adv = tf.equal(self.eval_preds, self.y_target) # one-to-one comparison
		if loss == 'cw':
			tlab = tf.one_hot(self.y, num_labels, on_value=1.0, off_value=0.0, dtype=tf.float32)
			target_probs = tf.reduce_sum(tlab*self.eval_probs, 1)
			other_probs = tf.reduce_max((1-tlab)*self.eval_probs - (tlab*10000), 1)
			self.loss = tf.log(other_probs + 1e-30) - tf.log(target_probs + 1e-30)
		elif loss == 'xent':
			self.loss = self.model.y_xent
		else:
			raise NotImplementedError

	# def __del__(self):
	#     self.sess.close()
	#     print('session close')

	def pred_class(self, data,batch_size = 1):
		if len(data.shape) > 2:
			data = data.reshape((-1,784)) 
		feed_dict = {self.x: data+self.bias}
		preds = self.sess.run(self.eval_preds, feed_dict)
		return preds

	def predict_prob(self, data, batch_size = 1):
		if len(data.shape) > 2:
			data = data.reshape((-1,784)) 
		feed_dict = {self.x: data+self.bias}
		probs = self.sess.run(self.eval_probs, feed_dict = feed_dict) 
		return probs # shape=(?,10)

	def get_loss(self, data, labels,class_num = 10):
		if len(data.shape) > 2:
			data = data.reshape((-1,784)) 
		feed_dict = {self.x: data+self.bias, self.y: labels}
		loss_val = self.sess.run(self.loss, feed_dict = feed_dict) 
		return loss_val

	def get_corr_pred_idx(self, all_images_o, all_labels):
		"""find the correctly classfied image index of given data and model in MNIST data"""
		if len(all_images_o.shape) > 2:
			all_images = np.copy(all_images_o)
			all_images = all_images.reshape((-1,784)) 
		num_eval_examples = len(all_images)
		eval_batch_size = 100
		num_batches = int(math.ceil(num_eval_examples / eval_batch_size))
		corr_pred_val = []
		for ibatch in range(num_batches):
			bstart = ibatch * eval_batch_size
			bend = min(bstart + eval_batch_size, num_eval_examples)
			x_batch = all_images[bstart:bend, :]
			y_batch = all_labels[bstart:bend]
			feed_dict= {self.x: x_batch+self.bias, self.y: y_batch}
			corr_pred_val.append(self.sess.run(self.correct_prediction, feed_dict = feed_dict))
		corr_pred_val = np.concatenate(corr_pred_val, axis=0)
		corr_pred_idx = np.nonzero(corr_pred_val)
		return corr_pred_idx[0]
	
	def eval_adv(self, adv, target_class):
		if len(adv.shape) > 2:
			adv = adv.reshape((-1,784))  
		feed_dict = {self.x: adv+self.bias, self.y_target: target_class}
		padv = self.sess.run(self.eval_percent_adv, feed_dict=feed_dict)
		return padv


class ZicoModel(object):
	def __init__(self, num_labels=10, model_dir="models/mnist.pth", bias = 0.0, loss='cw'):
		self.model = zico()
		self.bias = bias
		# load model weight
		self.model.load_state_dict(torch.load(model_dir))
		self.model.eval()
		self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
		self.model.to(self.device)
		self.loss = loss
		self.num_labels = num_labels

	def __del__(self):
		pass


	def pred_class(self, data,eval_batch_size = 1):
		# data.shape = (?, 784)       
		data = torch.from_numpy(data)
		data = data.float()
		data = data.view(-1, 1, 28, 28)
		data = data.to(self.device)
		num_eval_examples = len(data)
		num_batches = int(math.ceil(num_eval_examples / eval_batch_size))
		preds = []
		for ibatch in range(num_batches):
			bstart = ibatch * eval_batch_size
			bend = min(bstart + eval_batch_size, num_eval_examples)
			x_batch = data[bstart:bend, :]
			y_batch = data[bstart:bend]
			# forward
			logits = self.model(x_batch+self.bias)
			#print(logits)
			_, predicted = logits.max(1)
			preds.append(predicted.cpu().numpy()) # shape = (?,))
		preds = np.concatenate(preds, axis=0)
		return preds

	def predict_prob(self, data,batch_size = 1):
		if len(data.shape) > 2:
			data = data.reshape((-1,784)) 
		data = torch.from_numpy(data)
		data = data.float()
		data = data.view(-1, 1, 28, 28)
		data = data.to(self.device)
		# forward
		logits = self.model(data+self.bias)
		probs = F.softmax(logits, dim=1)
		#print(probs)
		probs = probs.detach().cpu().numpy()
		return probs # shape=(?,10)

	def get_loss(self, data, labels,class_num = 10):
		if len(data.shape) > 2:
			data = data.reshape((-1,784)) 
		data = torch.from_numpy(data)
		data = data.float()
		data = data.view(-1, 1, 28, 28)
		data = data.to(self.device)
		if type(labels) == np.ndarray:
			labels = torch.from_numpy(labels)
		else:
			labels = torch.LongTensor(np.array([labels]))
		labels = labels.to(self.device)
		# forward
		logits = self.model(data+self.bias)
		if self.loss == 'cw':
			probs = F.softmax(logits, dim=1)
			label_one_hot = torch.zeros(labels.size()+(self.num_labels,))
			label_one_hot = label_one_hot.to(self.device)
			label_one_hot.scatter_(1, labels.unsqueeze(1), 1.)
			target_prob = (probs*label_one_hot).sum(1)
			other_prob = ( (1. - label_one_hot) * probs - label_one_hot * 10000. ).max(1)[0]
			loss = torch.log(other_prob + 1e-30) - torch.log(target_prob + 1e-30)
		elif self.loss == 'xent':
			xent = nn.CrossEntropyLoss()
			loss = xent(logits, labels)
		else:
			raise NotImplementedError
		if loss.size() == torch.Size([1]):
			return loss.item() # shape = None, float
		else:
			return loss.detach().cpu().numpy() # shape = (?,)

	def get_corr_pred_idx(self, all_images_o, all_labels,eval_batch_size = 100):
		"""find the correctly classfied image index of given data and model in MNIST data"""
		if len(all_images_o.shape) > 2:
			all_images = np.copy(all_images_o)
			all_images = all_images.reshape((-1,784)) 
		num_eval_examples = len(all_images)
		num_batches = int(math.ceil(num_eval_examples / eval_batch_size))
		corr_pred_val = []
		for ibatch in range(num_batches):
			bstart = ibatch * eval_batch_size
			bend = min(bstart + eval_batch_size, num_eval_examples)
			x_batch = all_images[bstart:bend, :]
			y_batch = all_labels[bstart:bend]
			preds = self.pred_class(x_batch+self.bias)
			corr_pred_val.append(preds == y_batch)
		corr_pred_val = np.concatenate(corr_pred_val, axis=0)
		corr_pred_idx = np.nonzero(corr_pred_val)
		return corr_pred_idx[0] # shape = (?,)
	
	def eval_adv(self, adv, target_class):
		# adv.shape = (?, 784)
		if len(adv.shape) > 2:
			adv = adv.reshape((-1,784))  
		adv = torch.from_numpy(adv)
		adv = adv.float()
		adv = adv.view(-1, 1, 28, 28)
		adv = adv.to(self.device)
		# forward
		logits = self.model(adv+self.bias)
		_, predicted = logits.max(1) # only one
		if predicted.item() == target_class:
			return 1
		else:
			return 0
