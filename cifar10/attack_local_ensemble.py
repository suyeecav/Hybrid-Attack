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
	def __init__(self, model_ls, epsilon, k, a, random_start, loss_func,targeted,robust_indx,kappa = 50,x =None,y = None,is_training=None,\
		keep_prob=None):
		"""Attack parameter initialization. The attack performs k steps of
			 size a, while always staying within epsilon from the initial
			 point.
			 :param: loss: a tensor of loss function of the original model
			 :param: kappa: confidence value for cw attack, original pgd
			 set to 50 for logit layer.
			 """
		self.model_ls = model_ls
		# define the placeholders used for calculating gradients...
		self.x = x
		self.y = y
		self.is_training = is_training
		self.keep_prob = keep_prob
		self.epsilon = epsilon
		self.k = k
		self.a = a
		self.rand = random_start
		self.targeted = targeted
		self.scale = 255
		self.bias = 0.5
		self.robust_indx = robust_indx
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

	def attack(self, x_nat, y, sess, plot_ite = 10,clip_min = 0,clip_max = 1,candi_sel='succ_model',candi_thres=1,sel_criteria = None):
		"""
		Given a set of examples (x_nat, y), returns a set of adversarial
		examples within epsilon of x_nat in l_infinity norm.
		:param: robust_idx: indicator of which model is a robust model and needs scaling
		:param: candi_thres: local candidate selection threshold: NO LONGER USED
		:param: sel_criteria: local candidate criteria: NO LONGER USED
		"""
		self.clip_max = clip_max
		self.clip_min = clip_min
		if self.rand:
			x = x_nat + np.random.uniform(-self.epsilon, self.epsilon, x_nat.shape)
		else:
			x = np.copy(x_nat)
		inst_num = len(x_nat)
		pgd_stp_cnt_mat = 1e10*np.ones((inst_num,len(self.model_ls)),dtype = int)
		orig_or_tar_class = np.argmax(y,axis = 1)
		for i in range(self.k):
			# print the info periodically
			self.print_info(sess,i,x,y,plot_ite)
			grad_np_list = []
			succ_model_num = np.zeros(inst_num, dtype = int) # store the number of models sucessfully attacked, for each instance
			for kk in range(len(self.grad_list)):
				grad_tensor = self.grad_list[kk]
				is_robust = self.robust_indx[kk]
				model = self.model_ls[kk]
				if is_robust:
					feed_dict={self.x: (x+self.bias)*self.scale, self.y: y,
					self.is_training:False,
					self.keep_prob:1}
					grad_np,preds = sess.run([grad_tensor,model.predictions], feed_dict=feed_dict)
					grad_np = grad_np/self.scale
				else:
					feed_dict={self.x: x, self.y: y,
					self.is_training:False,
					self.keep_prob:1}
					grad_np,preds = sess.run([grad_tensor,model.predictions], feed_dict=feed_dict)
				grad_np_list.append(grad_np)
				pred_class = np.argmax(preds,axis = 1)

				# count the number of steps of successful pgd attack
				if self.targeted:
					succ_model_num[pred_class == orig_or_tar_class] += 1
				else:
					succ_model_num[pred_class != orig_or_tar_class] += 1
			# update the pgd matrix
			for j in reversed(range(len(self.model_ls))):
				# find instances that can be updated in current successfully attacked model number
				update_idx1 = np.logical_and(succ_model_num >= (j + 1),succ_model_num > 0)
				update_idx2 =  pgd_stp_cnt_mat[:,j] > i
				update_idx = np.logical_and(update_idx1,update_idx2)
				pgd_stp_cnt_mat[update_idx,j] = i

			grad = sum(grad_np_list) / len(grad_np_list)
			assert type(grad) == np.ndarray
			# need to calculate each gradient and some up with its weight
			x += self.a * np.sign(grad)
			x = np.clip(x, x_nat - self.epsilon, x_nat + self.epsilon) 
			x = np.clip(x, clip_min, clip_max) # ensure valid pixel range
			if i == self.k -1:
				max_loss, min_loss, ave_loss, max_gap, \
				min_gap, ave_gap = self.get_candi_metric(sess,x,x_nat,y,sel_criteria)	
		return x, max_loss, min_loss, ave_loss, max_gap, min_gap, ave_gap, pgd_stp_cnt_mat

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
			feed_dict = {self.x:x,self.y:y,
			self.is_training:False,
			self.keep_prob:1}
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

	def generate_candidates(self,sess,x,x_nat,y,metric,candidate_thres,inter_x,inter_x_resume,not_visited):
		'''
		this function serves the purpose of generating intermediate x and \
		subsequent iterations from intermediate x
		'''
		k_resume = 40
		clip_min = self.clip_min 
		clip_max = self.clip_max
		x_cpy = np.copy(x)
		inst_num = len(x)
		min_loss = 1e10*np.ones(inst_num)
		min_confid = 1e10*np.ones(inst_num)
		succ_model_num = np.zeros(inst_num, dtype = int) 
		for j in range(len(self.model_ls)):
			model = self.model_ls[j]
			feed_dict = {self.x:x,self.y:y,
			self.is_training:False,
			self.keep_prob:1}
			loss, preds = sess.run([model.loss, model.predictions],feed_dict = feed_dict)
			real = []
			other = []
			for idx in range(len(y)):
				real.append(preds[idx,np.argmax(y[idx])])
				other.append(np.max(preds[idx,np.logical_not(y[idx])]))
			real = np.array(real)
			other = np.array(other)
			pred_class = np.argmax(preds,axis = 1)
			orig_or_tar_class = np.argmax(y,axis = 1)
			if metric == 'succ_model':
				# criteria is k out of K models sucessfully attacked
				if self.targeted:
					succ_model_num[pred_class == orig_or_tar_class] += 1
				else:
					succ_model_num[pred_class != orig_or_tar_class] += 1
			elif metric == 'min_loss':
				# criteria is to choose seeds smaller than a given threshold T
				min_loss = np.minimum(loss,min_loss)
			elif metric == 'min_confidence':
				if self.targeted:
					confid_diff = np.log(real) - np.log(other)
				else:
					confid_diff = np.log(other) - np.log(real)
				confid_diff = np.minimum(confid_diff,min_confid)
		if metric == 'succ_model':
			valid_candi_idx = np.logical_and(succ_model_num >= candidate_thres,not_visited)
		elif metric == 'min_loss':
			valid_candi_idx = np.logical_and(min_loss < candidate_thres,not_visited)
		elif metric == 'min_confidence':
			valid_candi_idx = np.logical_and(confid_diff >= candidate_thres,not_visited)

		if valid_candi_idx.any():
			not_visited[valid_candi_idx] = 0
			# update the inter_x and inter_x_resume
			inter_x[valid_candi_idx] = x_cpy[valid_candi_idx]
			inter_x_resume[valid_candi_idx] = x_cpy[valid_candi_idx]
			x_tmp = inter_x_resume[valid_candi_idx] + np.random.uniform(-self.epsilon, self.epsilon, x_nat[valid_candi_idx].shape)
			x_tmp = np.clip(x_tmp, x_nat[valid_candi_idx] - self.epsilon, x_nat[valid_candi_idx] + self.epsilon) 

			y_tmp = y[valid_candi_idx]
			for i in range(k_resume):
				grad_np_list = []
				for grad_tensor in self.grad_list:
					feed_dict = {self.x:x,self.y:y,
								self.is_training:False,
								self.keep_prob:1}
					grad_np_list.append(sess.run(grad_tensor, feed_dict=feed_dict))
				grad = sum(grad_np_list) / len(grad_np_list)
				x_tmp += self.a * np.sign(grad)
				x_tmp = np.clip(x_tmp, x_nat[valid_candi_idx] - self.epsilon, x_nat[valid_candi_idx] + self.epsilon) 
				x_tmp = np.clip(x_tmp, clip_min, clip_max) # ensure valid pixel range
			inter_x_resume[valid_candi_idx] = x_tmp
		return inter_x, inter_x_resume, not_visited

	def print_info(self,sess,i,x,y,plot_ite):
		'''
		this function serves as printing the intermediate results, for \
		debugging purpose
		'''
		if i % plot_ite == 0 or i == self.k-1:
			########## for printing purpose ###############
			for j in range(len(self.model_ls)):
				model = self.model_ls[j]
				is_robust = self.robust_indx[j]
				if is_robust:
					feed_dict = {self.x:(x+self.bias)*self.scale,self.y:y,
					self.is_training:False,
					self.keep_prob:1}
				else:
					feed_dict = {self.x:x,self.y:y,
					self.is_training:False,
					self.keep_prob:1}
				loss, preds = sess.run([model.loss, model.predictions],feed_dict = feed_dict)
				real = []
				other = []
				for idx in range(len(y)):
					real.append(preds[idx,np.argmax(y[idx])])
					other.append(np.max(preds[idx,np.logical_not(y[idx])]))
				real = np.array(real)
				other = np.array(other)
				pred_class = np.argmax(preds,axis = 1)
				orig_or_tar_class = np.argmax(y,axis = 1)
				# print('loss,real,other,pred_class:',loss.shape,real.shape,other.shape,pred_class.shape)
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
				


 