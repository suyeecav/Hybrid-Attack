"""
Implementation of attack methods. Running this file as a program will
apply the attack to the model specified by the config file and store
the examples in an .npy file.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import copy

class LinfPGDAttack:
	def __init__(self, model_ls, epsilon, k, a, random_start, loss_func,targeted,kappa = 50,x =None,y = None):
		"""Attack parameter initialization. The attack performs k steps of
			 size a, while always staying within epsilon from the initial
			 point.
			 :param: loss: a tensor of loss function of the original model
			 :param: kappa: confidence value for cw attack, original pgd
			 set to 50 for logit layer
			 """
		self.model_ls = model_ls

		# define the placeholders used for calculating gradients...
		self.x = x
		self.y = y
		self.epsilon = epsilon
		self.k = k
		self.a = a
		self.rand = random_start
		self.targeted = targeted

		self.grad_list = []
		self.attack_lost_list = []

		for model in model_ls:
			if loss_func == 'xent':
				attack_loss = model.loss
				if targeted:
					attack_loss = -attack_loss
			elif loss_func == 'cw':
				label_mask = self.y
				correct_logit = tf.reduce_sum(label_mask * model.predictions, axis=1)
				wrong_logit = tf.reduce_max((1-label_mask) * model.predictions, axis=1)
				if targeted:
					attack_loss = -tf.nn.relu(wrong_logit - correct_logit + kappa)
				else:
					attack_loss = -tf.nn.relu(correct_logit - wrong_logit + kappa)
			else:
				print('Unknown loss function. Defaulting to cross-entropy')
				attack_loss = model.loss
				if targeted:
					attack_loss = -attack_loss
			self.attack_lost_list.append(attack_loss)
			self.grad_list.append(tf.gradients(attack_loss, self.x)[0])

	def attack(self, x_nat, y, sess, sign_flag=True, plot_ite = 10,clip_min = 0,clip_max = 1):
		"""Given a set of examples (x_nat, y), returns a set of adversarial
			 examples within epsilon of x_nat in l_infinity norm.
			 """
		if self.rand:
			x = x_nat + np.random.uniform(-self.epsilon, self.epsilon, x_nat.shape)
		else:
			x = np.copy(x_nat)
		inst_num = len(x_nat)
		# inst*model_num matrix: stores the ite step to successfully attack some # of local models
		pgd_stp_cnt_mat = 1e10*np.ones((inst_num,len(self.model_ls)),dtype = int)
		grads = []
		x_s = []
		for i in range(self.k):
			succ_model_num = np.zeros(inst_num, dtype = int) # store the number of models sucessfully attacked, for each instance
			########## start of printing stats ###############
			for j in range(len(self.model_ls)):
				model = self.model_ls[j]
				loss, preds = sess.run([model.loss, model.predictions],feed_dict = {self.x:x,self.y:y})
				real = []
				other = []
				for idx in range(len(y)):
					real.append(preds[idx,np.argmax(y[idx])])
					other.append(np.max(preds[idx,np.logical_not(y[idx])]))
				real = np.array(real)
				other = np.array(other)
				pred_class = np.argmax(preds,axis = 1)
				orig_or_tar_class = np.argmax(y,axis = 1)
				if i % plot_ite == 0:
					print("[Debug Info][Iter {}] model{}, loss:{:.5f}, real:{:.5f}, other:{:.5f}, pred class:{}, orig_or_tar class: {}".format(i, j,\
											loss[0], real[0], other[0],pred_class[0],orig_or_tar_class[0]))
				if i == self.k-1:
					print("[Final Info][Iter {}] model{}, loss:{:.5f}, real:{:.5f}, other:{:.5f}, pred class:{}, orig_or_tar class: {}".format(i, j,\
											loss[0], real[0], other[0],pred_class[0],orig_or_tar_class[0]))
					if self.targeted:
						succ_rate = np.sum(pred_class == orig_or_tar_class)/len(orig_or_tar_class)
						print("Attack Success Rate of Model {} is: {}".format(j,succ_rate))
					else:
						succ_rate = np.sum(pred_class != orig_or_tar_class)/len(orig_or_tar_class)
						print("Attack Success Rate of Model {} is: {}".format(j,succ_rate))
				# count the number of steps of successful pgd attack
				if self.targeted:
					succ_model_num[pred_class == orig_or_tar_class] += 1
				else:
					succ_model_num[pred_class != orig_or_tar_class] += 1

			for j in reversed(range(len(self.model_ls))):
				# find instances that can be updated in current successfully attacked model number
				update_idx1 = np.logical_and(succ_model_num >= (j + 1),succ_model_num > 0)
				update_idx2 =  pgd_stp_cnt_mat[:,j] > i
				update_idx = np.logical_and(update_idx1,update_idx2)
				pgd_stp_cnt_mat[update_idx,j] = i

			##### start of printing stats ###############

			####### gradient averaging ##############
			grad_np_list = []
			for grad_tensor in self.grad_list:
				grad_np_list.append(sess.run(grad_tensor, feed_dict={self.x: x, self.y: y}))
			
			grad = sum(grad_np_list) / len(grad_np_list)
			assert type(grad) == np.ndarray

			######## start of main attack #############
			x += self.a * np.sign(grad)
			x = np.clip(x, x_nat - self.epsilon, x_nat + self.epsilon) 
			x = np.clip(x, clip_min, clip_max) # ensure valid pixel range
			if i == self.k -1:
				max_loss, min_loss, ave_loss, max_gap, \
				min_gap, ave_gap = self.get_candi_metric(sess,x,x_nat,y)	
			######## end of main attack #############
		
		return x, np.array(grads),np.array(x_s),pgd_stp_cnt_mat, max_loss, min_loss, ave_loss, max_gap, min_gap, ave_gap

	def get_candi_metric(self,sess,x,x_nat,y,sel_criteria = None):
		'''
		-- this function currently returns: 1: loss function of each value, 2: confidence value gap
		-- currently, return both the maximum, minimum and average value.  
		-- useful for prioritizing seeds based on local model information 
		'''
		max_loss = (-1e10) * np.ones(len(x))
		min_loss = (1e10) * np.ones(len(x))
		ave_loss = np.zeros(len(x))
		max_gap = (-1e10) * np.ones(len(x))
		min_gap = (1e10) * np.ones(len(x))
		ave_gap = np.zeros(len(x))
		for j in range(len(self.model_ls)):
			model = self.model_ls[j]
			feed_dict = {self.x:x,self.y:y}
			loss, preds = sess.run([model.loss, model.predictions],feed_dict = feed_dict)
			real = []
			other = []
			for idx in range(len(y)):
				real.append(preds[idx,np.argmax(y[idx])])
				other.append(np.max(preds[idx,np.logical_not(y[idx])]))
			real = np.array(real)
			other = np.array(other)
			if self.targeted:
				confidence_gap = real - other
				loss = -loss
			else:
				confidence_gap = other - real
			# update the corresponding metrics
			max_loss = np.maximum(max_loss,loss)
			min_loss = np.minimum(min_loss,loss)
			ave_loss += loss
			max_gap = np.maximum(max_gap,confidence_gap)
			min_gap = np.minimum(min_gap,confidence_gap)
			ave_gap += confidence_gap
		ave_loss = ave_loss/len(self.model_ls)
		ave_gap = ave_gap/len(self.model_ls)
		return max_loss, min_loss, ave_loss, max_gap, min_gap, ave_gap 