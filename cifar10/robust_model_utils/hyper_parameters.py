# -*- coding: utf-8 -*-

'''
  Copyright(c) 2018, LiuYang
  All rights reserved.
  2018/02/23
'''
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

## The following flags are related to save paths, tensorboard outputs and screen outputs
tf.app.flags.DEFINE_string('version', 'vgg', '''A version number defining the directory to save
logs and checkpoints''')
tf.app.flags.DEFINE_integer('report_freq', 391, '''Steps takes to output errors on the screen
and write summaries''')
tf.app.flags.DEFINE_float('train_ema_decay', 0.95, '''The decay factor of the train error's
moving average shown on tensorboard''')

tf.app.flags.DEFINE_string('train_path', 'summary_logs/train.csv',
                           ''' path of train data''')
tf.app.flags.DEFINE_string('vali_path', 'summary_logs/test.csv',
                           ''' path of validation data''')

## The following flags define hyper-parameters regards training

tf.app.flags.DEFINE_integer('train_steps', 90000, '''Total steps that you want to train''')
tf.app.flags.DEFINE_boolean('is_full_validation', True, '''Validation w/ full validation set or
a random batch''')
tf.app.flags.DEFINE_integer('train_batch_size', 128, '''Train batch size''')
tf.app.flags.DEFINE_integer('validation_batch_size', 250, '''Validation batch size, better to be
a divisor of 10000 for this task''')
tf.app.flags.DEFINE_integer('test_batch_size', 1000, '''Test batch size''')

tf.app.flags.DEFINE_float('init_lr', 0.1, '''Initial learning rate''')
tf.app.flags.DEFINE_float('lr_decay_factor', 0.1, '''How much to decay the learning rate each
time''')
tf.app.flags.DEFINE_integer('decay_step0', 40000, '''At which step to decay the learning rate''')
tf.app.flags.DEFINE_integer('decay_step1', 60000, '''At which step to decay the learning rate''')


## The following flags define hyper-parameters modifying the training network

tf.app.flags.DEFINE_integer('num_residual_blocks', 5, '''How many residual blocks do you want''')
tf.app.flags.DEFINE_float('weight_decay', 0.0002, '''scale for l2 regularization''')


## The following flags are related to data-augmentation

tf.app.flags.DEFINE_integer('padding_size', 2, '''In data augmentation, layers of zero padding on
each side of the image''')


## If you want to load a checkpoint and continue training

tf.app.flags.DEFINE_string('ckpt_path', 'summary_logs/logs_vgg', '''Checkpoint
directory to restore''')
tf.app.flags.DEFINE_boolean('is_use_ckpt', False, '''Whether to load a checkpoint and continue
training''')

tf.app.flags.DEFINE_string('test_ckpt_path', 'logs_vgg', '''Checkpoint
directory to restore''')


train_dir = 'logs_' + FLAGS.version + '/'
