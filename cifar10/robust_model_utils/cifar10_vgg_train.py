# -*- coding: utf-8 -*-

'''
  Copyright(c) 2018, LiuYang
  All rights reserved.
  2018/02/23
'''

from .vgg import *
from datetime import datetime
import time
# from cifar10_vgg_processing import *
# from pgd_attack import LinfPGDAttack
# import cifar10_input
import os
import math

############### some params needed ################
IMG_WIDTH = 32
IMG_HEIGHT = 32
IMG_DEPTH = 3
NUM_CLASS = 10

TRAIN_RANDOM_LABEL = False # Want to use random label for train data?
VALI_RANDOM_LABEL = False # Want to use random label for validation?

NUM_TRAIN_BATCH = 5 # How many batches of files you want to read in, from 0 to 5)
EPOCH_SIZE = 10000 * NUM_TRAIN_BATCH
####################################################

class model_wrap(object):
    '''
    this is just for the easiness of model for pgd attacks
    '''
    def __init__(self,xent,pre_softmax,x_input,y_input):
        self.xent = xent
        self.x_input = x_input
        self.y_input = y_input
        self.pre_softmax = pre_softmax

class Train(object):
    '''
    This Object is responsible for all the training and validation process
    '''
    def __init__(self):
        # Set up all the placeholders
        self.placeholders()

    def placeholders(self):
        '''
        There are five placeholders in total.
        image_placeholder and label_placeholder are for train images and labels
        vali_image_placeholder and vali_label_placeholder are for validation imgaes and labels
        lr_placeholder is for learning rate. Feed in learning rate each time of training
        implements learning rate decay easily
        '''
        self.x_input = tf.placeholder(dtype=tf.float32,
                                                shape=[FLAGS.train_batch_size, IMG_HEIGHT,
                                                        IMG_WIDTH, IMG_DEPTH])
        self.y_input = tf.placeholder(dtype=tf.int32, shape=[FLAGS.train_batch_size])

        self.vali_image_placeholder = tf.placeholder(dtype=tf.float32, shape=[FLAGS.validation_batch_size,
                                                                IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH])
        self.vali_label_placeholder = tf.placeholder(dtype=tf.int32, shape=[FLAGS.validation_batch_size])

        self.lr_placeholder = tf.placeholder(dtype=tf.float32, shape=[])

    def build_train_validation_graph(self):
        '''
        This function builds the train graph and validation graph at the same time.
        
        '''
        global_step = tf.Variable(0, trainable=False)
        validation_step = tf.Variable(0, trainable=False)

        # Logits of training data and valiation data come from the same graph. The inference of
        # validation data share all the weights with train data. This is implemented by passing
        # reuse=True to the variable scopes of train graph
        logits = inference_VGG(self.x_input, reuse=False)
        self.pre_softmax = logits
        vali_logits = inference_VGG(self.vali_image_placeholder, reuse=True)

        # The following codes calculate the train loss, which is consist of the
        # softmax cross entropy and the relularization loss
        regu_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        loss = self.loss(logits, self.y_input)
        self.xent = loss
        self.full_loss = tf.add_n([loss] + regu_losses)

        predictions = tf.nn.softmax(logits)
        self.train_top1_error = self.top_k_error(predictions, self.y_input, 1)

        # Validation loss
        self.vali_loss = self.loss(vali_logits, self.vali_label_placeholder)
        vali_predictions = tf.nn.softmax(vali_logits)
        self.vali_top1_error = self.top_k_error(vali_predictions, self.vali_label_placeholder, 1)

        self.train_op, self.train_ema_op = self.train_operation(global_step, self.full_loss,
                                                                self.train_top1_error)
        self.val_op = self.validation_op(validation_step, self.vali_top1_error, self.vali_loss)

    def train(self):
        '''
        This is the main function for training
        '''

        # For the first step, we are loading all training images and validation images into the
        # memory
        data_path = "cifar10_data"
        raw_cifar = cifar10_input.CIFAR10Data(data_path)
        vali_data, vali_labels = raw_cifar.eval_data.xs, raw_cifar.eval_data.ys
        all_data, all_labels = raw_cifar.train_data.xs, raw_cifar.train_data.ys
        print('x_train shape:', all_data.shape)
        print('y_train shape:', all_labels.shape)
        print('x_test shape:', vali_data.shape)
        print('y_test shape:', vali_labels.shape)
        print(all_data.shape[0], 'train samples')
        print(vali_data.shape[0], 'test samples')

        print("max pixel value:",np.max(all_data),np.max(vali_data))
        print("min pixel value",np.min(all_data),np.min(vali_data))

        # all_data, all_labels = read_in_all_images(filename=FLAGS.train_path, shuffle=True)
        # vali_data, vali_labels = read_in_all_images(filename=FLAGS.vali_path, shuffle=True)

        # Build the graph for train and validation
        self.build_train_validation_graph()

        # Initialize a saver to save checkpoints. Merge all summaries, so we can run all
        # summarizing operations by running summary_op. Initialize a new session
        saver = tf.train.Saver(var_list = tf.global_variables())
        summary_op = tf.summary.merge_all()
        init = tf.initialize_all_variables()
        sess = tf.Session()

        # define the model wrap and define the pgd attack graph
        model = model_wrap(self.xent,self.pre_softmax,self.x_input,self.y_input)
        attack = LinfPGDAttack(model,
                epsilon = 8.0,
                num_steps = 10,
                step_size = 2.0,
                random_start = True,
                loss_func = "xent")

        # If you want to load from a checkpoint
        if FLAGS.is_use_ckpt is True:
            ckpt = tf.train.get_checkpoint_state(FLAGS.ckpt_path)
            offset_step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
            print(offset_step)
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('Restored from checkpoint...')
        else:
            offset_step=0
            sess.run(init)

        # calculate total size of parameter
        print('----------------------------')
        Param = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]) / (1024 * 1024)
        print('Number of parameter: %f M' % Param)
        print('----------------------------')

        # This summary writer object helps write summaries on tensorboard
        summary_writer = tf.summary.FileWriter(train_dir, sess.graph)


        # These lists are used to save a csv file at last
        step_list = []
        train_error_list = []
        val_error_list = []

        print('Start training...')
        print('----------------------------')
        cifar = cifar10_input.AugmentedCIFAR10Data(raw_cifar, sess, model = 0)
        for step in range(offset_step, FLAGS.train_steps):

            # train_batch_data, train_batch_labels = self.generate_augment_train_batch(all_data, all_labels,
            #                                     FLAGS.train_batch_size)
            train_batch_data, train_batch_labels = cifar.train_data.get_next_batch(batch_size = FLAGS.train_batch_size,
														multiple_passes=True)
            # generate adversarial perturbation
            train_batch_data_adv = attack.perturb(train_batch_data, train_batch_labels, sess)

            # Want to validate once before training. You may check the theoretical validation
            # loss first
            if (step-1) % (FLAGS.report_freq*10) == 0 and step != 0:

                # validation_batch_data, validation_batch_labels = self.generate_vali_batch(vali_data,
                #                                                vali_labels, FLAGS.validation_batch_size)

                if FLAGS.is_full_validation is True:
                    validation_loss_value, validation_error_value = self.full_validation(loss=self.vali_loss,
                                            top1_error=self.vali_top1_error, vali_data=vali_data,
                                            vali_labels=vali_labels, session=sess,
                                            batch_data=train_batch_data, batch_label=train_batch_labels)

                    vali_summ = tf.Summary()
                    vali_summ.value.add(tag='full_validation_error',
                                        simple_value=validation_error_value.astype(np.float))
                    summary_writer.add_summary(vali_summ, step)
                    summary_writer.flush()

                # else:
                #     _, validation_error_value, validation_loss_value = sess.run([self.val_op,
                #                                                      self.vali_top1_error,
                #                                                  self.vali_loss],
                #                                 {self.x_input: train_batch_data,
                #                                  self.y_input: train_batch_labels,
                #                                  self.vali_image_placeholder: validation_batch_data,
                #                                  self.vali_label_placeholder: validation_batch_labels,
                #                                  self.lr_placeholder: FLAGS.init_lr})

                val_error_list.append(validation_error_value)

                print('Previous: Validation top1 error = %.4f' % validation_error_value)
                print('Previous: Validation loss = ', validation_loss_value)
            start_time = time.time()

            _, _, train_loss_value_adv, train_error_value_adv = sess.run([self.train_op, self.train_ema_op,
                                                           self.full_loss, self.train_top1_error],
                                {self.x_input: train_batch_data_adv,
                                  self.y_input: train_batch_labels,
                                  self.lr_placeholder: FLAGS.init_lr})

            # calculate the error value of natural images
            train_loss_value, train_error_value = sess.run([self.full_loss, self.train_top1_error],
                                {self.x_input: train_batch_data,
                                 self.y_input: train_batch_labels,
                                 self.lr_placeholder: FLAGS.init_lr})
            duration = time.time() - start_time

            if step % FLAGS.report_freq == 0:
                # validation_batch_data, validation_batch_labels = self.generate_vali_batch(vali_data,
                #                                                                           vali_labels,
                #                                                                           FLAGS.validation_batch_size)

                # summary_str = sess.run(summary_op, {self.x_input: train_batch_data,
                #                                     self.y_input: train_batch_labels,
                #                                     self.vali_image_placeholder: validation_batch_data,
                #                                     self.vali_label_placeholder: validation_batch_labels,
                #                                     self.lr_placeholder: FLAGS.init_lr})
                # summary_writer.add_summary(summary_str, step)

                num_examples_per_step = FLAGS.train_batch_size
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)

                format_str = ('%s: step %d, adversarial loss = %.4f (%.1f examples/sec; %.3f ' 'sec/batch)')
                print(format_str % (datetime.now(), step, train_loss_value_adv, examples_per_sec,
                                    sec_per_batch))
                print('Train top1 adversarial error = ', train_error_value_adv)
                print('Train top1 natural error = ', train_error_value)
                print('Train natural loss = ', train_loss_value)
                print('----------------------------')

                step_list.append(step)
                train_error_list.append(train_error_value)

            if step == FLAGS.decay_step0 or step == FLAGS.decay_step1:
                FLAGS.init_lr = 0.1 * FLAGS.init_lr
                print('Learning rate decayed to ', FLAGS.init_lr)

            # Save checkpoints every 10000 steps
            if step % 10000 == 0 or (step + 1) == FLAGS.train_steps:
                checkpoint_path = os.path.join(train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

                # df = pd.DataFrame(data={'step':step_list, 'train_error':train_error_list,
                #                 'validation_error': val_error_list})
                # df.to_csv(train_dir + FLAGS.version + '_error.csv')
    def test(self):
        '''
        This function is used to evaluate the test data. Please finish pre-precessing in advance

        :param test_image_array: 4D numpy array with shape [num_test_images, img_height, img_width,
        img_depth]
        :return: the softmax probability with shape [num_test_images, num_labels]
        '''
        # num_test_images = len(test_image_array)
        # num_batches = num_test_images // FLAGS.test_batch_size
        # remain_images = num_test_images % FLAGS.test_batch_size
        # print('%i test batches in total...' %num_batches)

        # Create the test image and labels placeholders
        self.test_image_placeholder = tf.placeholder(dtype=tf.float32, shape=[FLAGS.test_batch_size,
                                                        IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH])
        self.test_label_placeholder = tf.placeholder(dtype=tf.int32, shape=[FLAGS.test_batch_size])
        self.one_hot_lab = tf.one_hot(self.test_label_placeholder,
                              10,
                              on_value=1,
                              off_value=0,
                              dtype=tf.int32)
        # Build the test graph
        logits = inference_VGG(self.test_image_placeholder, reuse=False)
        self.pre_softmax = logits
        predictions = tf.nn.softmax(logits)

        # correct_prediction = tf.equal(tf.argmax(logits,1), self.one_hot_lab)
        # num_correct = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))
        # losses
        loss = self.loss(logits, self.test_label_placeholder)
        self.xent = loss


        # Initialize a new session and restore a checkpoint
        saver = tf.train.Saver()
        sess = tf.Session()

        # ckpt = tf.train.get_checkpoint_state(FLAGS.test_ckpt_path)
        ckpt = tf.train.latest_checkpoint(FLAGS.test_ckpt_path)
        if ckpt:
            # Restores from checkpoint
            print('Model restored from ', FLAGS.test_ckpt_path)
            saver.restore(sess, ckpt)
            # saver.restore(sess, ckpt.model_checkpoint_path)
            # check_step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
            # print('checkpoint step:', check_step)
        else:
            print('No checkpoint file found. Please check the checkpoint dir')
            return

        data_path = "cifar10_data"
        cifar = cifar10_input.CIFAR10Data(data_path)

        # Iterate over the samples batch-by-batch
        x_test, y_test = cifar.eval_data.xs, cifar.eval_data.ys
        num_eval_examples = len(x_test)
        eval_batch_size = FLAGS.test_batch_size
        num_batches = int(math.ceil(num_eval_examples / eval_batch_size))

        # first check the normal accuracy
        print('Iterating over {} batches'.format(num_batches))
        total_nat_corr = 0
        for ibatch in range(num_batches):
            bstart = ibatch * eval_batch_size
            bend = min(bstart + eval_batch_size, num_eval_examples)
            print('batch size: {}'.format(bend - bstart))
            x_batch = x_test[bstart:bend, :]
            y_batch = y_test[bstart:bend]
            preds = sess.run(predictions,
                                        feed_dict={self.test_image_placeholder: x_batch,
                                        self.test_label_placeholder: y_batch})
            corr_num = np.sum(np.argmax(preds,1) == y_batch)
            total_nat_corr += corr_num
        print("normal test accuracy is:",total_nat_corr/len(x_test))

        # define the attack graph
        model = model_wrap(self.xent,self.pre_softmax,self.test_image_placeholder,self.test_label_placeholder)
        attack = LinfPGDAttack(model,
                epsilon = 8.0,
                num_steps = 10,
                step_size = 2.0,
                random_start = True,
                loss_func = "xent")
        x_adv = [] # adv accumulator
        total_adv_corr = 0
        for ibatch in range(num_batches):
            bstart = ibatch * eval_batch_size
            bend = min(bstart + eval_batch_size, num_eval_examples)
            print('batch size: {}'.format(bend - bstart))

            x_batch = x_test[bstart:bend, :]
            y_batch = y_test[bstart:bend]

            x_batch_adv = attack.perturb(x_batch, y_batch, sess)
            preds = sess.run(predictions,
                                        feed_dict={self.test_image_placeholder: x_batch_adv,
                                        self.test_label_placeholder: y_batch})
            corr_num = np.sum(np.argmax(preds,1) == y_batch)
            total_adv_corr += corr_num
            x_adv.append(x_batch_adv)
        x_adv = np.concatenate(x_adv, axis=0)
        print("adversarial test accuracy is:",total_adv_corr/len(x_test))
        # prediction_array = np.array([]).reshape(-1, NUM_CLASS)
        # # Test by batches
        # for step in range(num_batches):
        #     if step % 10 == 0:
        #         print('%i batches finished!' %step)
        #     offset = step * FLAGS.test_batch_size
        #     test_image_batch = test_image_array[offset:offset+FLAGS.test_batch_size, ...]

        #     batch_prediction_array = sess.run(predictions,
        #                                 feed_dict={self.test_image_placeholder: test_image_batch})

        #     prediction_array = np.concatenate((prediction_array, batch_prediction_array))

        # # If test_batch_size is not a divisor of num_test_images
        # if remain_images != 0:
        #     self.test_image_placeholder = tf.placeholder(dtype=tf.float32, shape=[remain_images,
        #                                                 IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH])
        #     # Build the test graph
        #     logits = inference_VGG(self.test_image_placeholder, reuse=True)
        #     predictions = tf.nn.softmax(logits)

        #     test_image_batch = test_image_array[-remain_images:, ...]

        #     batch_prediction_array = sess.run(predictions, feed_dict={
        #         self.test_image_placeholder: test_image_batch})

        #     prediction_array = np.concatenate((prediction_array, batch_prediction_array))

        return total_nat_corr, total_adv_corr



    ## Helper functions
    def loss(self, logits, labels):
        '''
        Calculate the cross entropy loss given logits and true labels
        :param logits: 2D tensor with shape [batch_size, num_labels]
        :param labels: 1D tensor with shape [batch_size]
        :return: loss tensor with shape [1]
        '''
        labels = tf.cast(labels, tf.int64)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                       labels=labels, name='cross_entropy_per_example')
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
        return cross_entropy_mean


    def top_k_error(self, predictions, labels, k):
        '''
        Calculate the top-k error
        :param predictions: 2D tensor with shape [batch_size, num_labels]
        :param labels: 1D tensor with shape [batch_size, 1]
        :param k: int
        :return: tensor with shape [1]
        '''
        batch_size = predictions.get_shape().as_list()[0]
        in_top1 = tf.to_float(tf.nn.in_top_k(predictions, labels, k=1))
        num_correct = tf.reduce_sum(in_top1)
        return (batch_size - num_correct) / float(batch_size)

    def generate_vali_batch(self, vali_data, vali_label, vali_batch_size):
        '''
        If you want to use a random batch of validation data to validate instead of using the
        whole validation data, this function helps you generate that batch
        :param vali_data: 1D numpy array. path of vali_data
        :param vali_label: 1D numpy array
        :param vali_batch_size: int
        :return: 4D numpy array and 1D numpy array
        '''
        offset = np.random.choice(len(vali_label) - vali_batch_size, 1)[0]
        path_vali_batch_data = vali_data[offset:offset+vali_batch_size]
        vali_data_batch = []
        for i in range(vali_batch_size):
            temp = np.load(path_vali_batch_data[i])
            vali_data_batch.append(temp)
        vali_data_batch = np.array(vali_data_batch, dtype=np.float32)
        vali_data_batch = np.reshape(vali_data_batch, (vali_batch_size, 3, 32, 32))
        vali_data_batch = vali_data_batch.transpose(0, 2, 3, 1)
        vali_data_batch = whitening_image(vali_data_batch)

        vali_label_batch = vali_label[offset:offset + vali_batch_size]
        return vali_data_batch, vali_label_batch



    def generate_augment_train_batch(self, train_data, train_labels, train_batch_size):
        '''
        This function helps generate a batch of train data, and random crop, horizontally flip
        and whiten them at the same time
        :param train_data: path of train_data 1D numpy array
        :param train_labels: 1D numpy array
        :param train_batch_size: int
        :return: augmented train batch data and labels. 4D numpy array and 1D numpy array
        '''
        offset = np.random.choice(len(train_labels) - train_batch_size, 1)[0]
        path_batch_data = train_data[offset:offset+train_batch_size]
        batch_data = []
        for i in range(train_batch_size):
            temp = np.load(path_batch_data[i])
            batch_data.append(temp)
        batch_data = np.array(batch_data)
        batch_data = np.reshape(batch_data,(train_batch_size, 3, 32, 32))
        batch_data = batch_data.transpose(0, 2, 3, 1)
        batch_data = batch_data.astype(np.float32)
        pad_width = ((0, 0), (FLAGS.padding_size, FLAGS.padding_size), (FLAGS.padding_size, FLAGS.padding_size), (0, 0))
        batch_data = np.pad(batch_data, pad_width=pad_width, mode='constant', constant_values=0)
        batch_data = random_crop_and_flip(batch_data, padding_size=FLAGS.padding_size)
        batch_data = whitening_image(batch_data)
        batch_label = train_labels[offset:offset + FLAGS.train_batch_size]

        return batch_data, batch_label



    def train_operation(self, global_step, total_loss, top1_error):
        '''
        Defines train operations
        :param global_step: tensor variable with shape [1]
        :param total_loss: tensor with shape [1]
        :param top1_error: tensor with shape [1]
        :return: two operations. Running train_op will do optimization once. Running train_ema_op
        will generate the moving average of train error and train loss for tensorboard
        '''
        # Add train_loss, current learning rate and train error into the tensorboard summary ops
        tf.summary.scalar('learning_rate', self.lr_placeholder)
        tf.summary.scalar('train_loss', total_loss)
        tf.summary.scalar('train_top1_error', top1_error)

        # The ema object help calculate the moving average of train loss and train error
        ema = tf.train.ExponentialMovingAverage(FLAGS.train_ema_decay, global_step)
        train_ema_op = ema.apply([total_loss, top1_error])
        tf.summary.scalar('train_top1_error_avg', ema.average(top1_error))
        tf.summary.scalar('train_loss_avg', ema.average(total_loss))

        opt = tf.train.MomentumOptimizer(learning_rate=self.lr_placeholder, momentum=0.9)
        train_op = opt.minimize(total_loss, global_step=global_step)
        return train_op, train_ema_op


    def validation_op(self, validation_step, top1_error, loss):
        '''
        Defines validation operations
        :param validation_step: tensor with shape [1]
        :param top1_error: tensor with shape [1]
        :param loss: tensor with shape [1]
        :return: validation operation
        '''

        # This ema object help calculate the moving average of validation loss and error

        # ema with decay = 0.0 won't average things at all. This returns the original error
        ema = tf.train.ExponentialMovingAverage(0.0, validation_step)
        ema2 = tf.train.ExponentialMovingAverage(0.95, validation_step)


        val_op = tf.group(validation_step.assign_add(1), ema.apply([top1_error, loss]),
                          ema2.apply([top1_error, loss]))
        top1_error_val = ema.average(top1_error)
        top1_error_avg = ema2.average(top1_error)
        loss_val = ema.average(loss)
        loss_val_avg = ema2.average(loss)

        # Summarize these values on tensorboard
        tf.summary.scalar('val_top1_error', top1_error_val)
        tf.summary.scalar('val_top1_error_avg', top1_error_avg)
        tf.summary.scalar('val_loss', loss_val)
        tf.summary.scalar('val_loss_avg', loss_val_avg)
        return val_op


    def full_validation(self, loss, top1_error, session, vali_data, vali_labels, batch_data,
                        batch_label):
        '''
        Runs validation on all the  valdiation images
        :param loss: tensor with shape [1]
        :param top1_error: tensor with shape [1]
        :param session: the current tensorflow session
        :param vali_data: 4D numpy array
        :param vali_labels: 1D numpy array
        :param batch_data: 4D numpy array. training batch to feed dict and fetch the weights
        :param batch_label: 1D numpy array. training labels to feed the dict
        :return: float, float
        '''
        num_batches = 10000 // FLAGS.validation_batch_size
        order = np.random.choice(10000, num_batches * FLAGS.validation_batch_size)
        vali_data_subset = vali_data[order, ...]
        vali_labels_subset = vali_labels[order]

        loss_list = []
        error_list = []

        for step in range(num_batches):
            offset = step * FLAGS.validation_batch_size
            feed_dict = {self.x_input: batch_data, self.y_input: batch_label,
                self.vali_image_placeholder: vali_data_subset[offset:offset+FLAGS.validation_batch_size, ...],
                self.vali_label_placeholder: vali_labels_subset[offset:offset+FLAGS.validation_batch_size],
                self.lr_placeholder: FLAGS.init_lr}
            loss_value, top1_error_value = session.run([loss, top1_error], feed_dict=feed_dict)
            loss_list.append(loss_value)
            error_list.append(top1_error_value)

        return np.mean(loss_list), np.mean(error_list)

class wrap_as_vgg_model(object):
    def __init__(self,xs,ys):
        '''
        :param: xs: placeholder for input data, should be of shape (None,img_size,img_size,nchannel) 
        :param: ys: placeholder for input lable, should be of shape (None,class_num)
        '''

        self.test_image_placeholder = xs
        self.test_label_placeholder = tf.argmax(ys,axis = 1)

        # Build the test graph
        logits = inference_VGG(self.test_image_placeholder, reuse=False)
        self.pre_softmax = logits
        predictions = tf.nn.softmax(logits)
        self.softmax_prob = predictions
        # losses
        loss, mean_loss = self.loss(logits, self.test_label_placeholder)
        self.xent = loss
        correct_prediction = tf.equal(tf.argmax(predictions, 1), self.test_label_placeholder)
        self.correct_prediction = correct_prediction
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        self.accuracy = accuracy


        ## Helper functions
    def loss(self, logits, labels):
        '''
        Calculate the cross entropy loss given logits and true labels
        :param logits: 2D tensor with shape [batch_size, num_labels]
        :param labels: 1D tensor with shape [batch_size]
        :return: loss tensor with shape [1]
        '''
        labels = tf.cast(labels, tf.int64)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                       labels=labels, name='cross_entropy_per_example')
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
        return cross_entropy,cross_entropy_mean

if __name__ == '__main__':
    #maybe_download_and_extract()
    # Initialize the Train object
    train = Train()
    mode = "train" 
    ## Start the training session
    if mode == "train":
        train.train()
    else:
        train.test()

    '''
    ## start the testing session

    # test data processing session
    # read test file
    path = r'test_data\test_batch'
    fo = open(path, 'rb')
    dicts = pickle.load(fo, encoding='bytes')
    fo.close()
    data = np.array(dicts[b'data'], dtype=np.float32)
    label = np.array(dicts[b'labels'], dtype=np.int32)
    data = np.reshape(data, (len(label), 3, 32, 32))
    data = data.transpose(0, 2, 3, 1)
    data = whitening_image(data)

    predict_result = train.test(data)
    print(label)
    print(predict_result)
    '''








