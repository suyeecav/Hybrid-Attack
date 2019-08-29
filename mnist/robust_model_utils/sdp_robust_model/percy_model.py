
import tensorflow as tf

class Model(object):
    def __init__(self):
        self.x_input = tf.placeholder(tf.float32, shape = [None, 784])
        self.y_input = tf.placeholder(tf.int64, shape = [None])
        with tf.variable_scope("model_weights") as scope:
            # First fully connected layer 
            W_fc1 = weight_variable("W_fc1", [784, 500])
            b_fc1 = bias_variable("b_fc1", [500])
            # ReLU activation
            # Second layer
            W = tf.get_variable("W_fc2",   initializer = tf.truncated_normal([500, 10], stddev = 0.1))
            b = tf.get_variable("b_fc2", initializer=tf.zeros([10]))
            h_fc1 = tf.nn.relu(tf.matmul(self.x_input, W_fc1) + b_fc1)
            self.pre_softmax = tf.matmul(h_fc1, W) + b
        
        self.y_pred = tf.argmax(self.pre_softmax, 1)
        self.correct_prediction = tf.equal(self.y_pred, self.y_input)
        self.y_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.y_input, logits=self.pre_softmax)

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