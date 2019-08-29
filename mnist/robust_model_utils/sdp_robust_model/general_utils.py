"""
File contains helper functions for defining tensorflow layers
"""

import tensorflow as tf

def conv2d(x, W, s):
  """conv2d returns a 2d convolution layer with input stride."""
  return tf.nn.conv2d(x, W, strides=[1, s, s, 1], padding='SAME')


def max_pool_2x2(x):
  """max_pool_2x2 downsamples a feature map by 2X."""
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(name, shape):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  with tf.variable_scope(name):
    weight_variable = tf.get_variable("weight_variable", initializer=initial)

  return weight_variable

def bias_variable(name, shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  with tf.variable_scope(name):
    bias_variable = tf.get_variable("bias_variable", initializer=initial)

  return bias_variable


