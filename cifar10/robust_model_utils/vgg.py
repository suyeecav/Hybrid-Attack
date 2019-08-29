# -*- coding: utf-8 -*-

'''
  Copyright(c) 2018, LiuYang
  All rights reserved.
  2018/02/23
'''


# This is the vgg structure

import numpy as np
import tensorflow as tf
# from .hyper_parameters import *

BN_EPSILON = 0.001
WEIGHT_DECAY = 0.0002
def activation_summary(x):
    '''
    :param x: A Tensor
    :return: Add histogram summary and scalar summary of the sparsity of the tensor
    '''
    tensor_name = x.op.name
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def create_variables(name, shape, initializer=tf.contrib.layers.xavier_initializer(), is_fc_layer=False):
    '''
    :param name: A string. The name of the new variable
    :param shape: A list of dimensions
    :param initializer: User Xavier as default.
    :param is_fc_layer: Want to create fc layer variable? May use different weight_decay for fc
    layers.
    :return: The created variable
    '''
    
    ## TODO: to allow different weight decay to fully connected layer and conv layer
    if is_fc_layer is True:
        regularizer = tf.contrib.layers.l2_regularizer(scale=WEIGHT_DECAY)
    else:
        regularizer = tf.contrib.layers.l2_regularizer(scale=WEIGHT_DECAY)

    new_variables = tf.get_variable(name, shape=shape, initializer=initializer,
                                    regularizer=regularizer)
    return new_variables

def fc_layer(input_layer, num_output, is_relu=True):
    '''
    full connection layer
    :param input_layer: 2D tensor
    :param num_output: number of output layer
    :param is_relu: judge use activation function: relu
    :return: output layer, 2D tensor
    '''
    input_dim = input_layer.get_shape().as_list()[-1]
    fc_w = create_variables(name='fc_weights', shape=[input_dim, num_output], is_fc_layer=True,
                            initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
    fc_b = create_variables(name='fc_bias', shape=[num_output], initializer=tf.zeros_initializer())

    fc_result = tf.matmul(input_layer, fc_w) + fc_b
    if is_relu is True:
        return tf.nn.relu(fc_result)
    else:
        return fc_result


def fc_bn_layer(input_layer, num_output, is_relu=True):
    '''
    full connection layer
    :param input_layer: 2D tensor
    :param num_output: number of output layer
    :param is_relu: judge use activation function: relu
    :return: output layer, 2D tensor
    '''
    input_dim = input_layer.get_shape().as_list()[-1]
    fc_w = create_variables(name='fc_weights', shape=[input_dim, num_output], is_fc_layer=True,
                            initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
    fc_b = create_variables(name='fc_bias', shape=[num_output], initializer=tf.zeros_initializer())

    fc_result = tf.matmul(input_layer, fc_w) + fc_b
    fc_bn_layer = batch_fc_normalization_layer(fc_result, num_output)
    if is_relu is True:
        return tf.nn.relu(fc_bn_layer)
    else:
        return fc_bn_layer

def batch_fc_normalization_layer(input_layer, dimension):
    '''
    Helper function to do batch normalziation of full connection layer
    :param input_layer: 2D tensor
    :param dimension: input_layer.get_shape().as_list()[-1]. The depth of the 2D tensor
    :return: the 2D tensor after being normalized
    '''
    mean, variance = tf.nn.moments(input_layer, axes=[0])
    beta = tf.get_variable('beta', dimension, tf.float32,
                           initializer=tf.constant_initializer(0.0, tf.float32))
    gamma = tf.get_variable('gamma', dimension, tf.float32,
                            initializer=tf.constant_initializer(1.0, tf.float32))
    fc_bn_layer = tf.nn.batch_normalization(input_layer, mean, variance, beta, gamma, BN_EPSILON)

    return fc_bn_layer


def batch_normalization_layer(input_layer, dimension):
    '''
    Helper function to do batch normalziation
    :param input_layer: 4D tensor
    :param dimension: input_layer.get_shape().as_list()[-1]. The depth of the 4D tensor
    :return: the 4D tensor after being normalized
    '''
    mean, variance = tf.nn.moments(input_layer, axes=[0, 1, 2])
    beta = tf.get_variable('beta', dimension, tf.float32,
                               initializer=tf.constant_initializer(0.0, tf.float32))
    gamma = tf.get_variable('gamma', dimension, tf.float32,
                                initializer=tf.constant_initializer(1.0, tf.float32))
    bn_layer = tf.nn.batch_normalization(input_layer, mean, variance, beta, gamma, BN_EPSILON)

    return bn_layer

def conv_relu_layer(input_layer, filter_shape, stride):
    '''
    A helper function to conv and relu the input tensor sequentially
    :param input_layer: 4D tensor
    :param filter_shape: list. [filter_height, filter_width, filter_depth, filter_number]
    :param stride: stride size for conv
    :return: 4D tensor. Y = Relu(conv(X))
    '''
    filter = create_variables(name='conv_relu', shape=filter_shape)
    conv_layer = tf.nn.conv2d(input_layer, filter, strides=[1, stride, stride, 1], padding='SAME')
    output = tf.nn.relu(conv_layer)

    return output


def conv_bn_relu_layer(input_layer, filter_shape, stride):
    '''
    A helper function to conv, batch normalize and relu the input tensor sequentially
    :param input_layer: 4D tensor
    :param filter_shape: list. [filter_height, filter_width, filter_depth, filter_number]
    :param stride: stride size for conv
    :return: 4D tensor. Y = Relu(batch_normalize(conv(X)))
    '''

    out_channel = filter_shape[-1]
    filter = create_variables(name='conv_bn_relu', shape=filter_shape)

    conv_layer = tf.nn.conv2d(input_layer, filter, strides=[1, stride, stride, 1], padding='SAME')
    bn_layer = batch_normalization_layer(conv_layer, out_channel)

    output = tf.nn.relu(bn_layer)
    return output


def bn_relu_conv_layer(input_layer, filter_shape, stride):
    '''
    A helper function to batch normalize, relu and conv the input layer sequentially
    :param input_layer: 4D tensor
    :param filter_shape: list. [filter_height, filter_width, filter_depth, filter_number]
    :param stride: stride size for conv
    :return: 4D tensor. Y = conv(Relu(batch_normalize(X)))
    '''

    in_channel = input_layer.get_shape().as_list()[-1]

    bn_layer = batch_normalization_layer(input_layer, in_channel)
    relu_layer = tf.nn.relu(bn_layer)

    filter = create_variables(name='bn_relu_conv', shape=filter_shape)
    conv_layer = tf.nn.conv2d(relu_layer, filter, strides=[1, stride, stride, 1], padding='SAME')
    return conv_layer


def inference_VGG(input_tensor_batch, reuse, var_scope = "vgg19"):
    '''
    vgg network architecture
    :param input_tensor_batch: 4D tensor
    :param reuse: To build train graph, reuse=False. To build validation graph and share weights
    with train graph, resue=True
    :return: last layer in the network. Not softmax-ed
    '''
    with tf.variable_scope(var_scope):
        layers = []
        # input_standardized = tf.map_fn(lambda img: tf.image.per_image_standardization(img),
        #                                                                    input_tensor_batch)
        input_standardized = input_tensor_batch
        # block1
        with tf.variable_scope('conv1_1', reuse=reuse):
            # conv1_1 = conv_bn_relu_layer(input_tensor_batch, [3, 3, 3, 64], 1)
            conv1_1 = conv_bn_relu_layer(input_standardized, [3, 3, 3, 64], 1)
            activation_summary(conv1_1)
            layers.append(conv1_1)
        with tf.variable_scope('conv1_2', reuse=reuse):
            conv1_2 = conv_bn_relu_layer(conv1_1, [3, 3, 64, 64], 1)
            activation_summary(conv1_2)
            layers.append(conv1_2)
        with tf.name_scope('conv1_max_pool'):
            conv2 = tf.nn.max_pool(conv1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            activation_summary(conv2)
            layers.append(conv2)
        # block2
        with tf.variable_scope('conv2_1', reuse=reuse):
            conv2_1 = conv_bn_relu_layer(conv2, [3, 3, 64, 128], 1)
            activation_summary(conv2_1)
            layers.append(conv2_1)
        with tf.variable_scope('conv2_2', reuse=reuse):
            conv2_2 = conv_bn_relu_layer(conv2_1, [3, 3, 128, 128], 1)
            activation_summary(conv2_2)
            layers.append(conv2_2)
        with tf.name_scope('conv2_max_pool'):
            conv3 = tf.nn.max_pool(conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            activation_summary(conv3)
            layers.append(conv3)
        # block3
        with tf.variable_scope('conv3_1', reuse=reuse):
            conv3_1 = conv_bn_relu_layer(conv3, [3, 3, 128, 256], 1)
            activation_summary(conv3_1)
            layers.append(conv3_1)
        with tf.variable_scope('conv3_2', reuse=reuse):
            conv3_2 = conv_bn_relu_layer(conv3_1, [3, 3, 256, 256], 1)
            activation_summary(conv3_2)
            layers.append(conv3_2)
        with tf.variable_scope('conv3_3', reuse=reuse):
            conv3_3 = conv_bn_relu_layer(conv3_2, [3, 3, 256, 256], 1)
            activation_summary(conv3_3)
            layers.append(conv3_3)
        with tf.variable_scope('conv3_4', reuse=reuse):
            conv3_4 = conv_bn_relu_layer(conv3_3, [3, 3, 256, 256], 1)
            activation_summary(conv3_4)
            layers.append(conv3_4)
        with tf.name_scope('conv3_max_pool'):
            conv4 = tf.nn.max_pool(conv3_4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            activation_summary(conv4)
            layers.append(conv4)
        # block4
        with tf.variable_scope('conv4_1', reuse=reuse):
            conv4_1 = conv_bn_relu_layer(conv4, [3, 3, 256, 512], 1)
            activation_summary(conv4_1)
            layers.append(conv4_1)
        with tf.variable_scope('conv4_2', reuse=reuse):
            conv4_2 = conv_bn_relu_layer(conv4_1, [3, 3, 512, 512], 1)
            activation_summary(conv4_2)
            layers.append(conv4_2)
        with tf.variable_scope('conv4_3', reuse=reuse):
            conv4_3 = conv_bn_relu_layer(conv4_2, [3, 3, 512, 512], 1)
            activation_summary(conv4_3)
            layers.append(conv4_3)
        with tf.variable_scope('conv4_4', reuse=reuse):
            conv4_4 = conv_bn_relu_layer(conv4_3, [3, 3, 512, 512], 1)
            activation_summary(conv4_4)
            layers.append(conv4_4)
        with tf.name_scope('conv4_max_pool'):
            conv5 = tf.nn.max_pool(conv4_4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            activation_summary(conv5)
            layers.append(conv5)
        # block5
        with tf.variable_scope('conv5_1', reuse=reuse):
            conv5_1 = conv_bn_relu_layer(conv5, [3, 3, 512, 512], 1)
            activation_summary(conv5_1)
            layers.append(conv5_1)
        with tf.variable_scope('conv5_2', reuse=reuse):
            conv5_2 = conv_bn_relu_layer(conv5_1, [3, 3, 512, 512], 1)
            activation_summary(conv5_2)
            layers.append(conv5_2)
        with tf.variable_scope('conv5_3', reuse=reuse):
            conv5_3 = conv_bn_relu_layer(conv5_2, [3, 3, 512, 512], 1)
            activation_summary(conv5_3)
            layers.append(conv5_3)
        with tf.variable_scope('conv5_4', reuse=reuse):
            conv5_4 = conv_bn_relu_layer(conv5_3, [3, 3, 512, 512], 1)
            activation_summary(conv5_4)
            layers.append(conv5_4)
        with tf.name_scope('conv5_max_pool'):
            conv6 = tf.nn.max_pool(conv5_4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            activation_summary(conv6)
            layers.append(conv6)
        # full connection layer

        fc_shape = conv6.get_shape().as_list()
        nodes = fc_shape[1]*fc_shape[2]*fc_shape[3]
        fc_reshape = tf.reshape(conv6, (fc_shape[0], nodes), name='fc_reshape')
        
        # fc6
        with tf.variable_scope('fc6', reuse=reuse):
            fc6 = fc_bn_layer(fc_reshape, 4096)
            activation_summary(fc6)
            layers.append(fc6)
        with tf.name_scope('dropout1'):
            fc6_drop = tf.nn.dropout(fc6, 0.5)
            activation_summary(fc6_drop)
            layers.append(fc6_drop)
        # fc7
        with tf.variable_scope('fc7', reuse=reuse):
            fc7 = fc_bn_layer(fc6_drop, 4096)
            activation_summary(fc7)
            layers.append(fc7)
        with tf.name_scope('dropout2'):
            fc7_drop = tf.nn.dropout(fc7, 0.5)
            activation_summary(fc7_drop)
            layers.append(fc7_drop)
        # fc8
        with tf.variable_scope('fc8', reuse=reuse):
            fc8 = fc_bn_layer(fc7_drop, 10, is_relu=False)
            activation_summary(fc8)
            layers.append(fc8)
    return layers[-1]



def test_graph(train_dir='logs'):
    '''
    Run this function to look at the graph structure on tensorboard. A fast way!
    :param train_dir:
    '''
    input_tensor = tf.constant(np.ones([128, 32, 32, 3]), dtype=tf.float32)
    result = inference_VGG(input_tensor, reuse=False)
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)
    summary_writer = tf.train.SummaryWriter(train_dir, sess.graph)
