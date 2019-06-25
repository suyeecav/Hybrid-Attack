## blackbox_attack.py -- attack a black-box network optimizing for l_2 distance using different methods: zoo, zoo_ae, zoo_rv and autozoom
## Copyright (C) 2019, Fnu Suya <fs5xz@virginia.edu>
## Copyright (C) 2018, Chun-Chen Tu <timtu@Umich.edu>
##                     PaiShun Ting <paishun@umich.edu>
##                     Pin-Yu Chen <Pin-Yu.Chen@ibm.com>
## Copyright (C) 2017, Huan Zhang <ecezhang@ucdavis.edu>.
## Copyright (C) 2016, Nicholas Carlini <nicholas@carlini.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.

import sys
import os
import tensorflow as tf
import numpy as np
import scipy.misc
import math
import time

LEARNING_RATE = 2e-3     # larger values converge faster to less accurate results

# settings for ADAM solver
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999

# @jit(nopython=True)
# def coordinate_ADAM(losses, indice, grad, hess, batch_size, mt_arr, vt_arr, real_modifier, up, down, lr, adam_epoch, beta1, beta2, proj):
def coordinate_ADAM(losses, indice, grad, hess, batch_size, mt_arr, vt_arr, real_modifier, lr, adam_epoch, beta1, beta2, proj):
    for i in range(batch_size):
        grad[i] = (losses[i*2+1] - losses[i*2+2]) / 0.0002 
    # ADAM update
    mt = mt_arr[indice]
    mt = beta1 * mt + (1 - beta1) * grad
    mt_arr[indice] = mt
    vt = vt_arr[indice]
    vt = beta2 * vt + (1 - beta2) * (grad * grad)
    vt_arr[indice] = vt
    # epoch is an array; for each index we can have a different epoch number
    epoch = adam_epoch[indice]
    corr = (np.sqrt(1 - np.power(beta2,epoch))) / (1 - np.power(beta1, epoch))
    m = real_modifier.reshape(-1)
    old_val = m[indice] 
    old_val -= lr * corr * mt / (np.sqrt(vt)  + 1e-8 )

    m[indice] = old_val
    adam_epoch[indice] = epoch + 1

def ADAM(losses, indice, grad, hess, batch_size, mt_arr, vt_arr, real_modifier, lr, adam_epoch, beta1, beta2, proj, beta, z, q=1):
    for i in range(q):
        grad[i] = q*(losses[i+1] - losses[0])* z[i] / beta

    # argument indice should be removed for the next version
    # the entire modifier is updated for every epoch and thus indice is not required


    avg_grad = np.mean(grad, axis=0)
    # ADAM update
    mt = mt_arr[indice]
    mt = beta1 * mt + (1 - beta1) * avg_grad
    mt_arr[indice] = mt
    vt = vt_arr[indice]
    vt = beta2 * vt + (1 - beta2) * (avg_grad * avg_grad)
    vt_arr[indice] = vt

    epoch = adam_epoch[indice]
    corr = (np.sqrt(1 - np.power(beta2,epoch))) / (1 - np.power(beta1, epoch))
    m = real_modifier.reshape(-1)
    old_val = m[indice] 
    old_val -= lr * corr * mt / (np.sqrt(vt)  + 1e-8 )
    m[indice] = old_val
    adam_epoch[indice] = epoch + 1

class blackbox_attack:
    def __init__(self, sess, model, args,num_channels,image_size,num_labels):
        # data information
        self.num_channels = num_channels
        self.image_size = image_size
        self.num_labels = num_labels
        self.model = model
        # attack settings
        self.sess = sess
        self.MAX_ITER = args["max_iterations"]
        self.PRINT_EVERY = args["print_every"]
        self.SWITCH_ITER = args["switch_iterations"]
        self.INIT_CONST = args["init_const"]
        self.ATTACK_TYPE = args["attack_type"]
        self.USE_TANH = args["use_tanh"]
        self.BATCH_SIZE = args["batch_size"]
        self.LEARNING_RATE = args["lr"]
        self.CONFIDENCE = args["confidence"]
        self.modifier_size = args["img_resize"]

        self.C = args["cost_threshold"]
        self.query_num = self.MAX_ITER * 2

        self.image_shape = (self.image_size, self.image_size, self.num_channels)
        self.modifier_shape = (self.modifier_size, self.modifier_size, self.num_channels)

        self.set_img_modifier()

        # true image
        self.timg = tf.Variable(np.zeros(self.image_shape), dtype=tf.float32)
        self.timg1 = tf.Variable(np.zeros(self.image_shape), dtype=tf.float32)#this variable stores the original variable
        # true label
        self.tlab = tf.Variable(np.zeros(self.num_labels), dtype=tf.float32)
        self.const = tf.Variable(0.0, dtype=tf.float32)

        # operations to assign information
        self.assign_timg = tf.placeholder(tf.float32, self.image_shape)
        self.assign_timg1 = tf.placeholder(tf.float32, self.image_shape) # host the original image
        self.assign_tlab = tf.placeholder(tf.float32, self.num_labels)
        self.assign_const = tf.placeholder(tf.float32)

        if args['dist_metric'] == 'l2':
            if self.USE_TANH:
                self.newimg = tf.tanh(self.img_modifier + self.timg)/2
                #normalize the tensor with its l2 norm
                self.newimg = tf.tanh(self.timg1)/2 + self.C*tf.nn.l2_normalize(self.newimg - tf.tanh(self.timg1)/2,axis = [1,2,3])
            else:
                self.modifier_up = 0.5 - self.timg
                self.modifier_down = -0.5 - self.timg
                cond1 = tf.cast(tf.greater(self.img_modifier, self.modifier_up), tf.float32)
                cond2 = tf.cast(tf.less_equal(self.img_modifier, self.modifier_up), tf.float32)
                cond3 = tf.cast(tf.greater(self.img_modifier, self.modifier_down), tf.float32)
                cond4 = tf.cast(tf.less_equal(self.img_modifier, self.modifier_down), tf.float32)
                self.img_modifier = tf.multiply(cond1, self.modifier_up) + tf.multiply(tf.multiply(cond2, cond3), self.img_modifier) + tf.multiply(cond4, self.modifier_down)

                self.newimg = self.img_modifier + self.timg
                self.newimg = self.timg1 + self.C*tf.nn.l2_normalize(self.newimg - self.timg1,axis = [1,2,3])
        elif args['dist_metric'] == 'li':
            if self.USE_TANH:
                self.newimg = tf.tanh(self.img_modifier + self.timg)/2

                self.upper = tf.clip_by_value(tf.tanh(self.timg1)/2+tf.constant(self.C, tf.float32),tf.constant(-0.5, tf.float32),tf.constant(0.5, tf.float32))
                self.lower = tf.clip_by_value(tf.tanh(self.timg1)/2-tf.constant(self.C, tf.float32),tf.constant(-0.5, tf.float32),tf.constant(0.5, tf.float32))
                self.newimg = tf.clip_by_value(self.newimg,self.lower,self.upper)

            else:
                self.modifier_up = 0.5 - self.timg
                self.modifier_down = -0.5 - self.timg
                cond1 = tf.cast(tf.greater(self.img_modifier, self.modifier_up), tf.float32)
                cond2 = tf.cast(tf.less_equal(self.img_modifier, self.modifier_up), tf.float32)
                cond3 = tf.cast(tf.greater(self.img_modifier, self.modifier_down), tf.float32)
                cond4 = tf.cast(tf.less_equal(self.img_modifier, self.modifier_down), tf.float32)
                self.img_modifier = tf.multiply(cond1, self.modifier_up) + tf.multiply(tf.multiply(cond2, cond3), self.img_modifier) + tf.multiply(cond4, self.modifier_down)
                self.newimg = self.img_modifier + self.timg
                #clip the instance to fit into Li ball.
                self.upper = tf.clip_by_value(self.timg1+tf.constant(self.C, tf.float32),tf.constant(-0.5, tf.float32),tf.constant(0.5, tf.float32))
                self.lower = tf.clip_by_value(self.timg1-tf.constant(self.C, tf.float32),tf.constant(-0.5, tf.float32),tf.constant(0.5, tf.float32))
                self.newimg = tf.clip_by_value(self.newimg,self.lower,self.upper)

        # the modifier beging updated and used
        print(self.modifier_shape)
        self.real_modifier = np.zeros((1,) + self.modifier_shape, dtype=np.float32)

        # self.output = model.predict(self.newimg)
        self.output = tf.placeholder(tf.float32, [None,self.num_labels])

        # distortion to the input data
        if args['dist_metric'] == 'l2':
            if self.USE_TANH:
                self.lp_dist = tf.reduce_sum(tf.square(self.newimg-tf.tanh(self.timg1)/2), [1,2,3])
            else:
                self.lp_dist = tf.reduce_sum(tf.square(self.newimg - self.timg1), [1,2,3])

            self.real = tf.reduce_sum((self.tlab)*self.output,1)
            self.other = tf.reduce_max((1-self.tlab)*self.output - (self.tlab*10000),1)

            if self.ATTACK_TYPE == "targeted":
                loss1 = tf.maximum(0.0, tf.log(self.other + 1e-30) - tf.log(self.real + 1e-30))
            elif self.ATTACK_TYPE == "untargeted":
                loss1 = tf.maximum(0.0, tf.log(self.real + 1e-30) - tf.log(self.other + 1e-30))

            # sum up the losses (output is a vector of # batch_size)
            self.loss2 = self.lp_dist
            self.loss1 = loss1
            self.loss = self.loss1
        elif args['dist_metric'] == 'li':
            # distortion to the input data
            if self.USE_TANH:
                self.lp_dist = tf.reduce_max(tf.abs(self.newimg-tf.tanh(self.timg1)/2),[1,2,3])
            else:
                self.lp_dist = tf.reduce_max(tf.abs(self.newimg - (self.timg1)),[1,2,3])

            self.real = tf.reduce_sum((self.tlab)*self.output,1)
            self.other = tf.reduce_max((1-self.tlab)*self.output - (self.tlab*10000),1)

            if self.ATTACK_TYPE == "targeted":
                loss1 = tf.maximum(0.0, tf.log(self.other + 1e-30) - tf.log(self.real + 1e-30))
            elif self.ATTACK_TYPE == "untargeted":
                loss1 = tf.maximum(0.0, tf.log(self.real + 1e-30) - tf.log(self.other + 1e-30))

            # sum up the losses (output is a vector of # batch_size)
            self.loss2 = self.lp_dist
            #self.loss1 = self.const*loss1
            self.loss1 = loss1
            self.loss = self.loss1
        
        # these are the variables to initialize when we run
        self.setup = []
        self.setup.append(self.timg.assign(self.assign_timg))
        self.setup.append(self.timg1.assign(self.assign_timg1))
        self.setup.append(self.tlab.assign(self.assign_tlab))
        self.setup.append(self.const.assign(self.assign_const))

        # prepare the list of all valid variables
        self.var_size = self.modifier_size * self.modifier_size * self.num_channels
        self.use_var_len = self.var_size
        self.var_list = np.array(range(0, self.use_var_len), dtype = np.int32)
        # ADAM status
        self.mt = np.zeros(self.var_size, dtype = np.float32)
        self.vt = np.zeros(self.var_size, dtype = np.float32)

        self.beta1 = ADAM_BETA1
        self.beta2 = ADAM_BETA2
        self.adam_epoch = np.ones(self.var_size, dtype = np.int32)

        self.stage = 0

    def set_img_modifier(self):
        print("set_up_img in blackbox is excuted")
        pass

    def print_info(self):
        pass

    def get_eval_cost(self):
        pass

    def compare(self,x,y):
        temp_x = np.copy(x)
        if not isinstance(x, (float, int, np.int64)):
            if self.ATTACK_TYPE == "targeted":
                temp_x[y] -= self.CONFIDENCE
                temp_x = np.argmax(temp_x)
            else:
                for i in range(len(temp_x)):
                    if i != y:
                        temp_x[i] -= self.CONFIDENCE
                temp_x = np.argmax(temp_x)
        if self.ATTACK_TYPE == "targeted":
            return temp_x == y
        else:
            return temp_x != y

    def attack(self, img, lab,orig_img):
        """
        Perform the L_ attack on the given images for the given targets.
        If self.targeted is true, then the targets represents the target labels.
        If self.targeted is false, then targets are the original class labels.
        """
        # remove the extra batch dimension

        C = self.C
        self.query_num = self.MAX_ITER * 2# initiliaze as reaching maximum number of iterations...
        
        #store label by-products
        x_s = []

        #just to avoid having o_bestattack in arctanh space
        o_bestattack = orig_img

        if len(img.shape) == 4:
            img = img[0]
            orig_img = orig_img[0]
        if len(lab.shape) == 2:
            lab = lab[0]
        # convert to tanh-space
        if self.USE_TANH:
            img = np.arctanh(img*1.999999)
            orig_img = np.arctanh(orig_img*1.999999)
        # set the lower and upper bounds accordingly
        lower_bound = 0.0
        CONST = self.INIT_CONST
        self.current_const = CONST
        upper_bound = 1e10
        # convert img to float32 to avoid numba error
        img = img.astype(np.float32)
        orig_img = orig_img.astype(np.float32)

        # set the upper and lower bounds for the modifier
        if not self.USE_TANH:
            self.modifier_up = 0.5 - img.reshape(-1)
            self.modifier_down = -0.5 - img.reshape(-1)

        # clear the modifier
        self.real_modifier.fill(0.0)

        # the over all best l2, score, and image attack
        o_bestl2 = 1e10
        o_bestscore = -1
        last_loss1 = 1e10

        # inner best l2 and scores
        bestl2 = 1e10
        bestscore = -1

        # setup the variablesnewimg_tmp
        self.sess.run(self.setup, {self.assign_timg: img,
                                       self.assign_tlab: lab,
                                       self.assign_const: CONST,
                                       self.assign_timg1: orig_img})

        prev = 1e6
        self.train_timer = 0.0
        last_loss2 = 1e10

        # reset ADAM status
        self.mt.fill(0.0)
        self.vt.fill(0.0)
        self.adam_epoch.fill(1)
        self.stage = 0

        self.eval_costs = 0
        attack_begin_time = time.time()
        # np.random.seed(1234)
        for iteration in range(self.MAX_ITER):
            if iteration % self.PRINT_EVERY == 0:
                self.current_iter = iteration
                self.print_info()
            if iteration  == 0:
                nimgs = self.sess.run(self.newimg,feed_dict = {self.modifier:self.real_modifier})
                output_prob = self.model.predict_prob(nimgs)
                feed_dict = {self.output: output_prob,self.modifier: self.real_modifier}
                loss, l2,real, other, loss1, loss2 = self.sess.run((self.loss,self.lp_dist,
                    self.real,self.other,self.loss1,self.loss2), feed_dict=feed_dict)

            # perform the attack 
            l, l2, loss1, loss2, score, nimg, real, other = self.blackbox_optimizer(iteration)
            self.eval_costs += self.get_eval_costs()

            if loss1 == 0.0 and last_loss1 != 0.0 and self.stage == 0:
                # we have reached the fine tunning point
                # reset ADAM to avoid overshoot
                print("##### Reset ADAM #####")
                self.mt.fill(0.0)
                self.vt.fill(0.0)
                self.adam_epoch.fill(1)
                self.stage = 1

            #store related important info
            x_s.append(nimg)
            if l2 < bestl2 and self.compare(score, np.argmax(lab)):
                bestl2 = l2
                bestscore = np.argmax(score)
            #terminate when feature modification is below the threshold...
            if l2 < C + C/100 and loss1 == 0.0:
                print("Early stopping because valid sample is found within threshold with {} queries".format(self.eval_costs))
                o_bestattack = nimg  # also updating the value of images and const
                o_best_const = CONST
                break

            if l2 < o_bestl2 and self.compare(score, np.argmax(lab)):
                # print a message if it is the first attack found
                if o_bestl2 == 1e10:
                    print("[STATS][FirstAttack] iter:{}, const:{}, cost:{}, time:{:.3f}, size:{}, loss:{:.5g}, loss1:{:.5g}, loss2:{:.5g}, l2:{:.5g}".format(iteration, CONST, self.eval_costs, self.train_timer, self.real_modifier.shape, l, loss1, loss2, l2))
                    self.post_success_setting()
                    lower_bound = 0.0
                o_bestl2 = l2
                o_bestscore = np.argmax(score)
                o_bestattack = nimg

            self.train_timer += time.time() - attack_begin_time

            last_loss1 = loss1
            last_loss2 = loss2
            # switch constant when reach switch iterations
            if (iteration+1) % self.SWITCH_ITER == 0:
                if self.compare(bestscore, np.argmax(lab)) and bestscore != -1:
                    # success, divide const by two
                    print("iter:{} old constant:{}".format(iteration, CONST))
                    upper_bound = min(upper_bound,CONST)
                    if upper_bound < 1e9:
                        CONST = (lower_bound + upper_bound)/2
                    print("iter:{} new constant:{}".format(iteration, CONST))
                else:
                    # failure, either multiply by 10 if no solution found yet
                    #          or do binary search with the known upper bound
                    print("iter:{} old constant:{}".format(iteration, CONST))
                    lower_bound = max(lower_bound,CONST)
                    if upper_bound < 1e9:
                        CONST = (lower_bound + upper_bound)/2
                    elif CONST < 1e8:
                        CONST *= 10
                    else:
                        print("CONST < 1e8, no change")
                    
                    print("iter:{} new constant:{}".format(iteration, CONST))

                bestl2 = 1e10
                bestscore = -1
                self.current_const = CONST

                if self.stage == 1:
                    self.mt.fill(0.0)
                    self.vt.fill(0.0)
                    self.adam_epoch.fill(1)

                # update constant
                self.sess.run(self.setup, {self.assign_timg: img,
                                       self.assign_tlab: lab,
                                       self.assign_const: CONST,
                                       self.assign_timg1: orig_img})
            # return the best solution found       
        return o_bestattack,self.eval_costs, x_s

    def print_info(self):
        nimgs = self.sess.run(self.newimg,feed_dict = {self.modifier:self.real_modifier})
        output_prob = self.model.predict_prob(nimgs)
        feed_dict = {self.output: output_prob,self.modifier: self.real_modifier}
        loss, real, other, loss1, loss2, o_const = self.sess.run((self.loss,self.real,self.other,self.loss1,self.loss2, self.const), feed_dict= feed_dict)
        print("[Info][Iter] iter:{}, const:{}, cost:{}, time:{:.3f}, size:{}, loss:{:.5g}, real:{:.5g}, other:{:.5g}, loss1:{:.5g}, loss2:{:.5g}".format(self.current_iter, self.current_const, self.eval_costs, self.train_timer, self.real_modifier.shape, loss[0], real[0], other[0], loss1[0], loss2[0]))
        sys.stdout.flush()

    def post_success_setting(self):
        pass

class ZOO(blackbox_attack):
    def __init__(self, sess, model, args,num_channels,image_size,num_labels):
        super().__init__(sess, model, args,num_channels,image_size,num_labels);
        self.grad = np.zeros(self.BATCH_SIZE, dtype = np.float32)
        self.hess = np.zeros(self.BATCH_SIZE, dtype = np.float32)
        self.solver = coordinate_ADAM

    def set_img_modifier(self):
        self.modifier = tf.placeholder(tf.float32, shape=(None,) + self.modifier_shape)
        if (self.modifier_size == self.image_size):
            # not resizing image or using autoencoder
            self.img_modifier = self.modifier
        else:
            # resizing image
            self.img_modifier = tf.image.resize_images(self.modifier, [self.image_size, self.image_size])

    def get_eval_costs(self):
        return self.BATCH_SIZE*2

    def blackbox_optimizer(self, iteration):
        # argument iteration is for debugging
        # build new inputs, based on current variable value
        var = np.repeat(self.real_modifier, self.BATCH_SIZE * 2 + 1, axis=0)
        var_size = self.real_modifier.size
        var_indice = np.random.choice(self.var_list.size, self.BATCH_SIZE, replace=False)
        indice = self.var_list[var_indice]
        for i in range(self.BATCH_SIZE):
            var[i * 2 + 1].reshape(-1)[indice[i]] += 0.0001
            var[i * 2 + 2].reshape(-1)[indice[i]] -= 0.0001
        losses, l2s, loss1, loss2, scores, nimgs,real, other = self.sess.run([self.loss, self.lp_dist, self.loss1, self.loss2, self.output, self.newimg,self.real,self.other], feed_dict={self.modifier: var})
        
        self.solver(losses, indice, self.grad, self.hess, self.BATCH_SIZE, self.mt, self.vt, self.real_modifier, self.LEARNING_RATE, self.adam_epoch, self.beta1, self.beta2, not self.USE_TANH)

        return losses[0], l2s[0], loss1[0], loss2[0], scores[0], nimgs[0], real[0],other[0]


class ZOO_AE(ZOO):
    def __init__(self, sess, model, args, decoder,num_channels,image_size,num_labels):
        self.decoder = decoder
        super().__init__(sess, model, args,num_channels,image_size,num_labels);

    def set_img_modifier(self):
        self.modifier = tf.placeholder(tf.float32, shape=(None,) + self.modifier_shape)
        if self.decoder.output_shape[1] == self.image_size:
            self.img_modifier = self.decoder(self.modifier)
        else:
            self.decoder_output =  self.decoder(self.modifier)
            self.img_modifier = tf.image.resize_images(self.decoder_output, [self.image_size, self.image_size])

class ZOO_RV(blackbox_attack):
    def __init__(self, sess, model, args,num_channels,image_size,num_labels):
        super().__init__(sess, model, args,num_channels,image_size,num_labels);
        
        self.hess = np.zeros(self.BATCH_SIZE, dtype = np.float32)
        self.solver = ADAM
        self.num_rand_vec = 1
        self.post_success_num_rand_vec = args["num_rand_vec"]
        self.grad = np.zeros((self.num_rand_vec, self.var_size), dtype = np.float32)

    def set_img_modifier(self):
        self.modifier = tf.placeholder(tf.float32, shape=(None,) + self.modifier_shape)
        if (self.modifier_size == self.image_size):
            # not resizing image or using autoencoder
            self.img_modifier = self.modifier
        else:
            # resizing image
            self.img_modifier = tf.image.resize_images(self.modifier, [self.image_size, self.image_size], align_corners=True)

    def get_eval_costs(self):
        return self.num_rand_vec + 1

    def blackbox_optimizer(self, iteration):

        # argument iteration is for debugging

        var_size = self.real_modifier.size
        indice = list(range(var_size))
        self.beta = 1/(var_size)

        var_noise = np.random.normal(loc=0, scale=1000, size=(self.num_rand_vec, var_size))
        var_mean = np.mean(var_noise, axis=1, keepdims=True)
        var_std = np.std(var_noise, axis=1, keepdims=True)

        noise_norm = np.apply_along_axis(np.linalg.norm, 1, var_noise, keepdims=True)
        var_noise = var_noise/noise_norm
        var = np.concatenate((self.real_modifier, self.real_modifier + self.beta*var_noise.reshape(self.num_rand_vec, self.modifier_size, self.modifier_size, self.num_channels)), axis=0)
        nimgs = self.sess.run(self.newimg,feed_dict = {self.modifier:var})
        # CHANGE IT WHEN POSSIBLE
        output_prob = self.model.predict_prob(nimgs)

        feed_dict = {self.output: output_prob, self.modifier: var}
        losses, l2s, loss1, loss2, scores, nimgs, real, other = self.sess.run([self.loss, self.lp_dist, self.loss1, self.loss2, self.output, self.newimg,self.real,self.other], feed_dict=feed_dict) 

        self.solver(losses, indice, self.grad, self.hess, self.BATCH_SIZE, self.mt, self.vt, self.real_modifier, self.LEARNING_RATE, self.adam_epoch, self.beta1, self.beta2, not self.USE_TANH, self.beta, var_noise, self.num_rand_vec)

        return losses[0], l2s[0], loss1[0], loss2[0], scores[0], nimgs[0], real[0], other[0]

    def post_success_setting(self):
        self.num_rand_vec = self.post_success_num_rand_vec
        self.grad = np.zeros((self.num_rand_vec, self.var_size), dtype = np.float32)
        print("Set random vector number to :{}".format(self.num_rand_vec))


class AutoZOOM(ZOO_RV):
    """
    def __init__(self, sess, model, args, decoder):
        
        self.decoder = decoder
        super().__init__(sess, model, args)
        self.num_rand_vec = 1
        self.post_success_num_rand_vec = args["num_rand_vec"]
        self.grad = np.zeros((self.num_rand_vec, self.var_size), dtype = np.float32)
        self.hess = np.zeros(self.BATCH_SIZE, dtype = np.float32)
        self.solver = ADAM
        """
    def __init__(self, sess, model, args, decoder, codec,num_channels,image_size,num_labels):
        self.codec = codec
        self.decoder = decoder
        super().__init__(sess, model, args,num_channels,image_size,num_labels)
        self.num_rand_vec = 1
        self.post_success_num_rand_vec = args["num_rand_vec"]
        self.grad = np.zeros((self.num_rand_vec, self.var_size), dtype = np.float32)
        self.hess = np.zeros(self.BATCH_SIZE, dtype = np.float32)
        self.solver = ADAM

    def set_img_modifier(self):
        """
        self.modifier = tf.placeholder(tf.float32, shape=(None,) + self.modifier_shape)
        self.img_modifier = self.decoder(self.modifier)
        """
        self.modifier = tf.placeholder(tf.float32, shape=(None,) + self.modifier_shape)

        if self.decoder.output_shape[1] == self.image_size:
            self.img_modifier = self.decoder(self.modifier)
        else:
            self.decoder_output =  self.decoder(self.modifier)
            self.img_modifier = tf.image.resize_images(self.decoder_output, [self.image_size, self.image_size], align_corners=True)


    def post_success_setting(self):
        self.num_rand_vec = self.post_success_num_rand_vec
        self.grad = np.zeros((self.num_rand_vec, self.var_size), dtype = np.float32)
        print("Set random vector number to :{}".format(self.num_rand_vec))

    def get_eval_costs(self):
        return self.num_rand_vec + 1



















