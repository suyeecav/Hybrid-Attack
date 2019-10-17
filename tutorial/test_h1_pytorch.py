import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader
from torch.nn import DataParallel
from torch.nn.modules import Upsample
from torch.autograd import Variable
import torch.optim as optim

from simBA import simba_single

import argparse
import json
import pdb
import numpy as np
import os

class LinfPGDAttack(object):
	def __init__(self, models=None, epsilon=0.3, k=40, a=0.01, 
		random_start=True, targeted=True):
		"""
		Attack parameter initialization. The attack performs k steps of
		size a, while always staying within epsilon from the initial
		point.
		https://github.com/MadryLab/mnist_challenge/blob/master/pgd_attack.py
		"""
		self.models = models
		self.epsilon = epsilon
		self.k = k
		self.a = a
		self.rand = random_start
		self.loss_fn = nn.CrossEntropyLoss()
		# TODO: add CW loss


	def perturb(self, X_nat, y):
		"""
		Given examples (X_nat, y), returns adversarial
		examples within epsilon of X_nat in l_infinity norm.
		"""
		if self.rand:
			X = X_nat + torch.distributions.uniform.Uniform(-self.epsilon, self.epsilon).sample()
		else:
			X = X_nat.clone().detach()


		def normalized_eval(x, model):
			normalized_x = torch.stack([F.normalize(x[i], [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) \
						for i in range(len(x))])
			return model(normalized_x)


		X_pgd = Variable(X.data, requires_grad=True)


		for i in range(self.k):
			print('local model attack iter:', i)
			grads_list = []
			loss_vals = []
			for model in self.models:
				# print('attacking model',  model.__class__.__name__)
				# zero gradient
				opt = optim.SGD([X_pgd], lr=1e-3)
				opt.zero_grad()
				model.cuda() # set to gpu 
				model = DataParallel(model) # set parallel
				with torch.enable_grad():
					logits = normalized_eval(X_pgd, model)
					loss = -self.loss_fn(logits, y)
				loss.backward()
				loss_vals.append(loss.item()) # for logging
				grads_list.append(X_pgd.grad.data.clone()) # for computing gradient
				# print success mask
				# success_mask = (logits.argmax(1) == y).float()
				# print(i, success_mask)
				# set model to cpu
				model.to('cpu') # to cpu to avoid gpu memory overflow
				torch.cuda.empty_cache() # clear cache
			# gradient averaging
			grads = torch.stack(grads_list)
			grad = torch.sum(grads, dim=0) / len(self.models)
			# update
			eta = self.a * grad.data.sign()
			X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
			eta = torch.clamp(X_pgd.data - X.data, -self.epsilon, self.epsilon) # eta [-epsilon, epsilon] range
			X_pgd = Variable(X.data + eta, requires_grad=True)
			X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True) # [0, 1] pixel range
			# logging
			print('loss: ', np.mean(loss_vals))
			torch.cuda.empty_cache()
			# break # TODO: delete

		return X_pgd.data.clone().detach()

CLASSIFIERS = {
	"inception_v3": (models.inception_v3, 299),
	"resnet50": (models.resnet50, 224),
	"vgg16_bn": (models.vgg16_bn, 224),
}

NUM_CLASSES = {
	"imagenet": 1000
}

# TODO: change the below to point to the ImageNet validation set,
# formatted for PyTorch ImageFolder
# Instructions for how to do this can be found at:
# https://github.com/facebook/fb.resnet.torch/blob/master/INSTALL.md#download-the-imagenet-dataset
IMAGENET_PATH = "/bigtemp/jc6ub/imagenet_torch/val"
if IMAGENET_PATH == "":
	raise ValueError("Please fill out the path to ImageNet")

def select_target_samples(model_to_fool, images, labels):
	'''select 100 examples for attack'''
	orig_classes = normalized_eval(images, model_to_fool).argmax(1)
	correct_classified_mask = (orig_classes == labels).float()
	corr_idx = torch.nonzero(correct_classified_mask)

	assert len(corr_idx) >= len(images) // 2 # args.data_loading_batch_size // 2

	corr_idx = corr_idx[:args.corr_batch_size].squeeze()
	images = images[corr_idx]
	labels = labels[corr_idx]
	targets = normalized_eval(images, model_to_fool).argmin(1)

	torch.cuda.empty_cache() # release gpu memory for unused variables

	return images, labels, targets

def normalized_eval(x, model):
		x_copy = x.clone()
		x_copy = torch.stack([F.normalize(x_copy[i], [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) \
						for i in range(len(x_copy))])
		return model(x_copy)


def get_probs(model, x, y):
	output = normalized_eval(x, model).cpu()
	probs = nn.Softmax(dim=1)(output)[:, y]
	return torch.diag(probs.data)

def get_preds(model, x):
	output = normalized_eval(x, model).cpu()
	_, preds = output.data.max(1)
	return preds

def attack_local_model_ensemble(images, targets, local_models):

	PGD = LinfPGDAttack(models=local_models, 
						epsilon=args.epsilon, 
						k=30,
						a=0.01, 
						random_start=True)

	images_adv = PGD.perturb(images, targets)

	return images_adv

def main(args, model_to_fool, dataset_size):
	dataset = ImageFolder(IMAGENET_PATH, 
					transforms.Compose([
						transforms.Resize(dataset_size),
						transforms.CenterCrop(dataset_size),
						transforms.ToTensor(),
					]))
	dataset_loader = DataLoader(dataset, batch_size=args.data_loading_batch_size, shuffle=True)
	total, total_adv, total_queries = 0, 0, 0
	success_flags, num_queries = [], []
	for i, (images, labels) in enumerate(dataset_loader):
		if i*args.data_loading_batch_size >= args.total_images:
			break

		# run the correct batch
		if i != args.batch_idx:
			continue
		print('running batch: ', i)
		############ local model attack ###########
		# step 1 select target label for correctly classfied images
		images, labels = images.cuda(), labels.cuda()
		images, labels, targets = select_target_samples(model_to_fool, images, labels)
		print(images.shape, labels.shape, targets.shape)
		# step 2 local model ensemble
		local_models = [models.resnet50(pretrained=True), 
						models.densenet121(pretrained=True),
						models.vgg19_bn(pretrained=True)]
		for model in local_models:
			model.eval()
		local_adv_images = attack_local_model_ensemble(images, targets, local_models)
		attack_seed = local_adv_images if args.attack_seed_type == 'adv' else images
		# calculate transfer rate
		orig_logits = normalized_eval(images, model_to_fool)
		adv_logits = normalized_eval(local_adv_images, model_to_fool)
		logits = adv_logits if args.attack_seed_type == 'adv' else orig_logits
		success_mask = (logits.argmax(1) == targets).float()
		print("transfer rate", torch.sum(success_mask)/len(success_mask))

		# step 3 black-box attack
		# attack image one by one
		for idx in range(len(images)):
			print('attacking image', idx)
			x = attack_seed[idx:idx+1]
			x_orig = images[idx:idx+1]
			y = targets[idx:idx+1]
			print('image target class', y)
			if args.black_attack == "simBA":
				x, queries, sucess_flag = simba_single(model_to_fool, x, y, x_orig, epsilon=args.epsilon)
			else:
				raise NotImplementedError
			num_queries.append(queries)
			success_flags.append(sucess_flag)

		# save query number and success
		fname = os.path.join(args.checkpoint, '{}_num_queries.txt'.format(args.attack_seed_type))
		np.savetxt(fname, num_queries)
		fname = os.path.join(args.checkpoint, '{}_success_flags.txt'.format(args.attack_seed_type))
		np.savetxt(fname, success_flags)

class Parameters():
	'''
	Parameters class, just a nice way of accessing a dictionary
	> ps = Parameters({"a": 1, "b": 3})
	> ps.A # returns 1
	> ps.B # returns 3
	'''
	def __init__(self, params):
		self.params = params
	
	def __getattr__(self, x):
		return self.params[x.lower()]

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--manual_seed', type=int, default=1234, help='manual seed')
	parser.add_argument('--max-queries',default=100000, type=int, help='max queries for each image')
	parser.add_argument('--tile-size', type=int, default=50, help='the side length of each tile (for the tiling prior)')
	parser.add_argument('--json-config', type=str, help='a config file to be passed in instead of arguments')
	parser.add_argument('--epsilon', type=float, default=12./255, help='the lp perturbation bound')
	parser.add_argument('--data-loading-batch-size', type=int, default=200, help='batch size for loading dataset')
	parser.add_argument('--checkpoint', type=str, default='checkpoint', help='checkpoint')
	parser.add_argument("--attack_seed_type", default="adv", choices=["orig", "adv"], help="attack seed type for batch attack")
	parser.add_argument('--total-images', type=int, default=1000)
	parser.add_argument('--classifier', type=str, default='vgg16_bn', choices=CLASSIFIERS.keys())
	parser.add_argument('--batch_idx', type=int, default=0)	
	parser.add_argument('--corr-batch-size', type=int, default=100, help='batch size for black-box attacks')
	parser.add_argument('--black-attack', type=str, default="simBA", help='for black-box attacks')

	args = parser.parse_args()

	torch.manual_seed(args.manual_seed)
	np.random.seed(args.manual_seed)
	torch.set_default_tensor_type('torch.cuda.FloatTensor')
	
	# save parameter 
	args.checkpoint = os.path.join(args.checkpoint, str(args.manual_seed))
	print('Creating checkpoint: ', args.checkpoint)
	if not os.path.exists(args.checkpoint):
		os.makedirs(args.checkpoint)

	args_dict = None
	if not args.json_config:
		# If there is no json file, all of the args must be given
		args_dict = vars(args)
	else:
		# If a json file is given, use the JSON file as the base, and then update it with args
		defaults = json.load(open(args.json_config))
		arg_vars = vars(args)
		arg_vars = {k: arg_vars[k] for k in arg_vars if arg_vars[k] is not None}
		defaults.update(arg_vars)
		args = Parameters(defaults)
		args_dict = defaults

	# save parameter in json file
	with open(os.path.join(args.checkpoint, 'args.json'), 'w') as f:
		json.dump(args.__dict__, f, indent=2)
	
	model_type = CLASSIFIERS[args.classifier][0]
	model_to_fool = model_type(pretrained=True)
	# set model in the right device and right mode
	model_to_fool = model_to_fool.cuda()
	model_to_fool = DataParallel(model_to_fool)
	model_to_fool.eval()

	with torch.no_grad():
		main(args, model_to_fool, CLASSIFIERS[args.classifier][1])