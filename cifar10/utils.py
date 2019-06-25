"""
general utility functions. Running this file will generate
inputs to the attacks and store in .npy files. Comment those if needed
"""
from cleverhans.utils_keras import KerasModelWrapper
import numpy as np
import tensorflow as tf
from PIL import Image
import math
from sklearn.metrics import accuracy_score

def save_img(img, img_size = 32,num_channel = 3,name = "output.png"):
	if len(img.shape) <=2:
		img = np.reshape(img,(-1,img_size,img_size,num_channel))
	# np.save(name, img)
	fig = np.around((img+0.5)*255)
	fig = fig.astype(np.uint8).squeeze()
	pic = Image.fromarray(fig)
	pic.save(name)

# directly wraps based on the model
class keras_model_wrapper():
	def __init__(self,model,x = None,y = None):
		# :param: x: placeholder for inputs
		# :param: y: placeholder for labels
		self.keras_model = model
		model_wrap = KerasModelWrapper(model)
		self.predictions = model_wrap.get_logits(x)
		self.probs = tf.nn.softmax(logits = self.predictions)
		self.loss = tf.nn.softmax_cross_entropy_with_logits(labels = y,
			logits = self.predictions) 

def generate_attack_inputs(sess,model,x_test,y_test,class_num,nb_imgs, load_imgs = False,load_robust = True,file_path = 'local_info/'):
    """ generate inputs to both the local and target models, including labels
    """
    if not load_imgs:
        # if load_robust:
        #     original_predict = sess.run(model.pre_softmax,feed_dict = {model.x_input:x_test.reshape(-1,784)})
        # else:
        original_predict = model.predict_prob(x_test)
        original_class = np.argmax(original_predict, axis = 1)
        true_class = np.argmax(y_test, axis = 1)
        mask = true_class == original_class
        corr_idx = np.where(true_class == original_class)[0]
        print(np.sum(mask), "out of", mask.size, "are correctly labeled,", len(x_test[mask]))   

        # generate targeted labels, choose least likely class, untargeted attack does not require
        orig_class_vec = range(class_num)
        target_ys_one_hot = []
        target_ys = []
        orig_images = []
        orig_labels = []
        all_true_ids = []
        trans_test_images = []
        for orig_class in orig_class_vec:
            # guarantees that same sets of images are selected...
            np.random.seed(1234)
            cls_idx = np.where(true_class == orig_class)[0]
            corr_cls_idx = np.intersect1d(cls_idx,corr_idx)
            #random sample, this is doubled because I want to set aside a test set which is to measure transferability
            corr_cls_idx = np.random.choice(corr_cls_idx, size=nb_imgs*2, replace=False)
            # print("selected img ids:",corr_cls_idx)
            x_sel = x_test[corr_cls_idx]
            orig_labels_tmp = y_test[corr_cls_idx]
            y_sel = model.predict_prob(x_sel)
            cls_count = np.bincount(np.argmin(y_sel,axis=1))# count number of occurence
            tar_class = np.argmax(cls_count)
            print("Orig: {}, Tar: {}".format(orig_class,tar_class))
            # store the adversarial samples related files
            target_ys.extend([tar_class]*int(len(corr_cls_idx)/2))
            target_ys_one_hot.extend([np.eye(class_num)[tar_class]]*int(len(corr_cls_idx)/2))
            orig_images.extend(instance for instance in x_sel[:int(len(x_sel)/2)])
            orig_labels.extend(lab for lab in orig_labels_tmp[:int(len(orig_labels_tmp)/2)])
            all_true_ids.extend(idx for idx in corr_cls_idx[:int(len(corr_cls_idx)/2)])
            # store the transferability test related files
            trans_test_images.extend(instance for instance in x_sel[int(len(x_sel)/2):])

        target_ys_one_hot = np.array(target_ys_one_hot)
        orig_images = np.array(orig_images)
        target_ys = np.array(target_ys)
        orig_labels = np.array(orig_labels)
        all_true_ids = np.array(all_true_ids)
        trans_test_images = np.array(trans_test_images)

        # can uncomment below if not needed
        fname = '/target_ys_one_hot_{}.npy'.format(nb_imgs)
        np.save(file_path+fname,target_ys_one_hot)
        fname = '/orig_images_{}.npy'.format(nb_imgs)
        np.save(file_path+fname,orig_images)
        fname = '/orig_labels_{}.npy'.format(nb_imgs)
        np.save(file_path+fname,orig_labels)
        fname = '/all_true_ids_{}.npy'.format(nb_imgs)
        np.save(file_path+fname,all_true_ids)
        fname = '/target_ys_{}.npy'.format(nb_imgs)
        np.save(file_path+fname,target_ys)
        fname = '/trans_test_images_{}.npy'.format(nb_imgs)
        np.save(file_path+fname,trans_test_images)
    else:
        fname = '/target_ys_one_hot_{}.npy'.format(nb_imgs)
        target_ys_one_hot = np.load(file_path+fname)
        fname = '/orig_images_{}.npy'.format(nb_imgs)
        orig_images = np.load(file_path+fname)
        fname = '/orig_labels_{}.npy'.format(nb_imgs)
        orig_labels = np.load(file_path+fname)
        fname = '/all_true_ids_{}.npy'.format(nb_imgs)
        all_true_ids = np.load(file_path+fname)
        fname = '/target_ys_{}.npy'.format(nb_imgs)
        target_ys = np.load(file_path+fname)
        fname = '/trans_test_images_{}.npy'.format(nb_imgs)
        trans_test_images = np.load(file_path+fname)
    return target_ys_one_hot,orig_images,target_ys,orig_labels,all_true_ids, trans_test_images

def compute_cw_loss(sess,model,data,labs,targeted = True,load_robust = True):
	""" CW loss will be calculated in this func
	:param: model: provide prediction scores
	:param: x: image instance, one hot form
	:param: y: specified instance
    """
	output = (model.predict_prob(data)) #shape:(?,class_num)
	real = np.sum(labs*output,axis = 1)
	other = np.amax((1-labs)*output - (labs*10000),1)
	if targeted:
		loss = np.maximum(0,np.log(other + 1e-30) - np.log(real + 1e-30))
		free_idx = np.where(real >= other)[0]
	else:
		loss = np.maximum(0,np.log(real + 1e-30) - np.log(other + 1e-30))
		free_idx = np.where(real <= other)[0]
	# print("free_idx:",free_idx)
	return loss, free_idx #shape (?,)


def select_next_seed(img_loss,attacked_flag,sort_metric, by_class,fine_tune_freq,class_num,per_cls_cnt,cls_order,change_limit,max_lim_num):
    """ generate the inputs for the local and target model attack. 
    param:sort_metric: if to select instances based on their loss value or randomly select seeds
    param:by_class: used for fine-tuning local models, if true, force to have equal number of seeds from each class (designed for improving fine-tuning, but FAILED!)
    """
    nb_imgs = int(len(img_loss)/class_num)
    if np.sum(np.logical_not(attacked_flag)) <= (len(img_loss) % fine_tune_freq):
        if not change_limit:
            max_lim_num =  int(np.sum(np.logical_not(attacked_flag))/class_num) # max instance num for each class
            change_limit = True

    if by_class:
        if fine_tune_freq % class_num != 0 or fine_tune_freq < class_num:
            print("if generate seeds by class, need to make update frequency dividble by class_num and is greater than class_num!")
            raise RuntimeError
        if (per_cls_cnt) % max_lim_num == 0 and per_cls_cnt !=0:
            print("max instance limit {} reached for class {}, reset to 0".format(per_cls_cnt,cls_order))
            per_cls_cnt = 0
            cls_order = (cls_order+1) % class_num
        # get the candidate index based on the loss function values
        img_loss_cls = img_loss[cls_order*nb_imgs:(cls_order+1)*nb_imgs]
        attacked_flag_cls = attacked_flag[cls_order*nb_imgs:(cls_order+1)*nb_imgs]
        if sort_metric == "min":
            candi_idx = np.argmin(img_loss_cls)
        elif sort_metric == "max":
            candi_idx = np.argmax(img_loss_cls)
        elif sort_metric == "random":
            idx_vec = np.array(range(len(img_loss_cls)))
            idx_vec = idx_vec[np.logical_not(attacked_flag_cls)]
            candi_idx = np.random.choice(idx_vec, size=1, replace=False)[0] # returns a list: [0]
        candi_idx = cls_order*nb_imgs+candi_idx
        per_cls_cnt += 1
    else:
        if sort_metric == "min":
            candi_idx = np.argmin(img_loss)
        elif sort_metric == "max":
            candi_idx = np.argmax(img_loss)
        elif sort_metric == "random":
            idx_vec = np.array(range(len(img_loss)))
            idx_vec = idx_vec[np.logical_not(attacked_flag)]
            candi_idx = np.random.choice(idx_vec, size=1, replace=False)[0] # returns a list: [0]
    return candi_idx, per_cls_cnt, cls_order, change_limit, max_lim_num 

def mixup_data(x,y,alpha = 1.0):
    """ planned to solve the fine-tuning problem with approach from
    paper: https://arxiv.org/abs/1710.09412, but failed. """

    if alpha > 0:
        lam = np.random.beta(alpha,alpha)
    else:
        lam = 1
    batch_size = len(x)
    index = np.random.permutation(batch_size)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    mixed_y = lam * y + (1-lam) * y[index]
    # y_a, y_b = y, y[index]
    return mixed_x, mixed_y, lam 

# evaluate the acuracy of models on batches
def local_attack_in_batches(sess, data,labels,eval_batch_size,attack_graph,model = None,clip_min = 0,clip_max = 1,load_robust=True):
	"""Iterate over the samples in batches to attack local models"""
	num_eval_examples = len(data)
	num_batches = int(math.ceil(num_eval_examples / eval_batch_size))
	X_test_adv = [] # adv accumulator
	pgd_cnt_mat = []
	max_losses = []
	min_losses = []
	ave_losses = []
	max_gaps = []
	min_gaps = []
	ave_gaps = []
	# succ_model_sum_s = []
	# print('Iterating over {} batches'.format(num_batches))
	for ibatch in range(num_batches):
		bstart = ibatch * eval_batch_size
		bend = min(bstart + eval_batch_size, num_eval_examples)
		# print('batch size: {}'.format(bend - bstart))
		x_batch = data[bstart:bend, :]
		y_batch = labels[bstart:bend,:]
		x_batch_adv_sub, max_loss, min_loss, ave_loss, max_gap, min_gap, ave_gap,\
			 pgd_stp_cnt_mat = attack_graph.attack(x_batch, y_batch, sess, clip_min = clip_min,clip_max = clip_max)
		# combine all results in all batches
		X_test_adv.extend(x_batch_adv_sub)
		pgd_cnt_mat.extend(pgd_stp_cnt_mat)
		max_losses.extend(max_loss)
		min_losses.extend(min_loss)
		ave_losses.extend(ave_loss)
		max_gaps.extend(max_gap)
		min_gaps.extend(min_gap)
		ave_gaps.extend(ave_gap)
		# succ_model_sum_s.extend(succ_model_sum)
	X_test_adv = np.array(X_test_adv)
	pgd_cnt_mat = np.array(pgd_cnt_mat)	
	max_gaps = np.array(max_gaps)
	min_gaps = np.array(min_gaps)
	ave_gaps = np.array(ave_gaps)
	max_losses = np.array(max_losses)
	min_losses = np.array(min_losses)
	ave_losses = np.array(ave_losses)
	# succ_model_sum_s = np.array(succ_model_sum_s)

	if model:
		pred_labs = np.argmax(model.predict_prob(np.array(X_test_adv)),axis=1)
		print('correct number',np.sum(pred_labs == np.argmax(labels,axis=1)))
		accuracy = accuracy_score(np.argmax(labels,axis=1), pred_labs)
	else:
		pred_labs = accuracy = 0
	return accuracy, pred_labs, X_test_adv, pgd_cnt_mat, max_losses, min_losses, ave_losses, max_gaps, min_gaps, ave_gaps