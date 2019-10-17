import torch
import torchvision.transforms.functional as F
import torch.nn as nn

# 20-line implementation of (targeted) SimBA for single image input
# code modify from https://github.com/cg563/simple-blackbox-attack/blob/master/simba_single.py
def simba_single(model, x, y, x_orig, epsilon, max_queries=100000, num_iters=100000, log_every=100):
	'''
	x_orig: original image
	x: attack_seed
	y:target class
	'''
	queries = 0

	n_dims = x.view(1, -1).size(1)
	perm = torch.randperm(n_dims)
	last_prob = get_probs(model, x, y)
	queries += 1
	sucess_flag = 0
	
	for i in range(num_iters):
		# check if attack succeed
		predicted_y = get_preds(model, x)
		if predicted_y.item() == y.item(): # equal to target class
			print("Attack succeed")
			sucess_flag = 1
			break

		if queries >= max_queries:
			print("Attack fails, achieving max queries")
			queries = max_queries
			break
				
		# craft left example
		x_left = x.clone()
		x_lower = x_orig.view(-1)[perm[i]] - epsilon
		x_left = x_left.view(-1)
		x_left[perm[i]] = x_lower
		x_left = x_left.view(x.size())

		left_prob = get_probs(model, x_left.clamp(0, 1), y)
		queries += 1

		if left_prob > last_prob:
			x = x_left.clamp(0, 1)
			last_prob = left_prob
		else:
			# craft right example
			x_right = x.clone()
			x_upper = x_orig.view(-1)[perm[i]] + epsilon
			x_right = x_right.view(-1)
			x_right[perm[i]] = x_upper
			x_right = x_right.view(x.size())

			right_prob = get_probs(model, x_right.clamp(0, 1), y)
			queries += 1
			if right_prob > last_prob:
				x = x_right.clamp(0, 1)
				last_prob = right_prob

		if (i + 1) % log_every == 0 or i == num_iters - 1:
			print('Iteration %d: queries = %d, prob = %.6f' % (
					i + 1, queries, last_prob.item()))

	return x, queries, sucess_flag

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
