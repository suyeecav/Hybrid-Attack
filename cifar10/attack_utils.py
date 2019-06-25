import numpy as np
import time
IMAGE_SIZE = (32,32,3)

def autozoom_attack(attack_graph,input_img,orig_img, label):
	"""run the autozoom style attacks on selected adversarial samples and record intermediate results
	:param: data: the image to be attacked
	param: label: the target or original label
	:return: 
	# x_s: intermediate results of the attack
	# ae, adversarial example
	# query_num: number of queries taken to attack the given seed
    """
	ae, query_num,x_s = attack_graph.attack(input_img, label, orig_img)
	return x_s, ae, query_num


def nes_attack(args,model,attack_seed,initial_img,target_class, class_num = 10, lower = -0.5, upper = 0.5):
	plot_ite = args["print_every"]
	max_lr = args["max_lr"]
	max_iters = int(np.ceil(args["max_queries"] // args["samples_per_draw"]))
	if args["attack_type"] == "targeted": 
		is_targeted = 1
	else:
		is_targeted  = -1
	lower = np.clip(initial_img - args["cost_threshold"], lower, upper)
	upper = np.clip(initial_img + args["cost_threshold"], lower, upper)
	adv = attack_seed.copy()

	# assert orig_class == model.pred_class(initial_img, axis=0), 'input image must be correctly classified'
	print('predicted class %d' % model.pred_class(initial_img))
	# HISTORY VARIABLES (for backtracking and momentum)
	num_queries = 0
	g = 0

	# adv = np.expand_dims(adv, axis=0) # wrap(unsqueeze) image to ensure 4-D np.array

	target_class = np.array([target_class]) # wrap(unsqueeze) labels to ensure 1-D np.array
	prev_adv = adv
	last_ls = []
	x_s = []
	# BEGIN ATTACK 
	# MAIN LOOP
	for i in range(max_iters):
		# record the intermediate attack results
		x_s.append(adv)
		##  ----------start the attack below-------- ##
		start = time.time()
		# GET GRADIENT
		prev_g = g
		l, g = get_grad_np(args, model, adv, target_class, args["samples_per_draw"], args["nes_batch_size"], \
			class_num,upper = upper,lower = lower)
		# print(l, g.shape)
		# SIMPLE MOMENTUM
		g = args["momentum"] * prev_g + (1.0 - args["momentum"]) * g
		# CALCULATE PROBABILITY
		eval_probs_val = model.predict_prob(adv)
		# CHECK IF WE SHOULD STOP
		padv = model.eval_adv(adv, target_class)
		predicted_class = model.pred_class(adv)
		if (padv == 1 and is_targeted == 1) or (padv == 0 and is_targeted == -1): # and epsilon <= goal_epsilon:
			print('[log] early stopping at iteration %d with query number %d' % (i,num_queries))
			print('[Final][Info]:predicted class: %d, target class: %d)' % (predicted_class,target_class))
			break
		# PLATEAU LR ANNEALING (if loss trend decreases in plateau_length,  then max_lr anneals)
		last_ls.append(l)
		last_ls = last_ls[-args["plateau_length"]:]
		if last_ls[-1] > last_ls[0] \
			and len(last_ls) == args["plateau_length"]:
			if max_lr > args["min_lr"]:
				print("[log] Annealing max_lr")
				max_lr = max(max_lr / args["plateau_drop"], args["min_lr"])
			last_ls = []

		# ATTACK
		current_lr = max_lr
		proposed_adv = adv - is_targeted * current_lr * np.sign(g) # (1,32,32,3) - (32,32,3) = (1,32,32,3)
		proposed_adv = np.clip(proposed_adv, lower, upper)
		prev_adv = adv
		adv = proposed_adv
		# BOOK-KEEPING STUFF
		num_queries += args["samples_per_draw"]
		predicted_class = model.pred_class(adv)
		if (i+1) % plot_ite == 0:
			log_text = 'Step %05d: number of query %d, loss %.8f, lr %.2E, predicted class %d target_class %d (time %.4f)' % (i, num_queries, l, \
				current_lr, predicted_class, target_class, time.time() - start)
			print(log_text)    
	return x_s, num_queries, adv

def get_grad_np(args,model, adv, tc, spd, bs, class_num,upper,lower):
    """ used for estimating gradients for NES, reimplemented in numpy
    """
    num_batches = spd // bs
    losses_val = []
    grads_val = []
    for _ in range(num_batches):
        tc = np.repeat(tc, bs) 
        noise_pos = np.random.normal(size=(bs//2,) + IMAGE_SIZE)
        noise = np.concatenate([noise_pos, -noise_pos], axis=0)
        eval_points = adv + args["sigma"] * noise * (upper-lower) # for scale
        loss_val = model.get_loss(eval_points, tc,class_num)
        losses_tiled = np.tile(np.reshape(loss_val, (-1, 1)), np.prod(IMAGE_SIZE))
        losses_tiled = np.reshape(losses_tiled, (bs,)+IMAGE_SIZE)
        grad_val = np.mean(losses_tiled * noise, axis=0)/args["sigma"]
        losses_val.append(loss_val)
        grads_val.append(grad_val)
    return np.array(losses_val).mean(), np.mean(np.array(grads_val), axis=0)

