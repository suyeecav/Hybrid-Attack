import numpy as np
import tensorflow as tf
import keras
# import cifar10_input
# from pgd_attack import LinfPGDAttack

def unpickle(file):
  import cPickle
  fo = open(file, 'rb')
  dict = cPickle.load(fo)
  fo.close()
  if 'data' in dict:
    dict['data'] = dict['data'].reshape((-1, 3, 32, 32)).swapaxes(1, 3).swapaxes(1, 2).reshape(-1, 32*32*3) / 256.

  return dict

def load_data_one(f):
  batch = unpickle(f)
  data = batch['data']
  labels = batch['labels']
  print ("Loading %s: %d" % (f, len(data)))
  return data, labels

def load_data(files, data_dir, label_count):
  data, labels = load_data_one(data_dir + '/' + files[0])
  for f in files[1:]:
    data_n, labels_n = load_data_one(data_dir + '/' + f)
    data = np.append(data, data_n, axis=0)
    labels = np.append(labels, labels_n, axis=0)
  labels = np.array([ [ float(i == label) for i in range(label_count) ] for label in labels ])
  return data, labels

def run_in_batch_avg(session, tensors, batch_placeholders, feed_dict={}, batch_size=200):                              
  res = [ 0 ] * len(tensors)                                                                                           
  batch_tensors = [ (placeholder, feed_dict[ placeholder ]) for placeholder in batch_placeholders ]                    
  total_size = len(batch_tensors[0][1])                                                                                
  batch_count = int((total_size + batch_size - 1) / batch_size)                                                             
  for batch_idx in range(batch_count):                                                                                
    current_batch_size = None                                                                                          
    for (placeholder, tensor) in batch_tensors:                                                                        
      batch_tensor = tensor[ batch_idx*batch_size : (batch_idx+1)*batch_size ]                                         
      current_batch_size = len(batch_tensor)                                                                           
      feed_dict[placeholder] = tensor[ batch_idx*batch_size : (batch_idx+1)*batch_size ]                               
    tmp = session.run(tensors, feed_dict=feed_dict)                                                                    
    res = [ r + t * current_batch_size for (r, t) in zip(res, tmp) ]                                                   
  return [ r / float(total_size) for r in res ]

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.01)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.01, shape=shape)
  return tf.Variable(initial)

def conv2d(input, in_features, out_features, kernel_size, with_bias=False):
  W = weight_variable([ kernel_size, kernel_size, in_features, out_features ])
  conv = tf.nn.conv2d(input, W, [ 1, 1, 1, 1 ], padding='SAME')
  if with_bias:
    return conv + bias_variable([ out_features ])
  return conv

def batch_activ_conv(current, in_features, out_features, kernel_size, is_training, keep_prob):
  current = tf.contrib.layers.batch_norm(current, scale=True, is_training=is_training, updates_collections=None)
  current = tf.nn.relu(current)
  current = conv2d(current, in_features, out_features, kernel_size)
  current = tf.nn.dropout(current, keep_prob)
  return current

def block(input, layers, in_features, growth, is_training, keep_prob):
  current = input
  features = in_features
  for idx in range(layers):
    tmp = batch_activ_conv(current, features, growth, 3, is_training, keep_prob)
    current = tf.concat((current, tmp), axis=3)
    features += growth
  return current, features

def avg_pool(input, s):
  return tf.nn.avg_pool(input, [ 1, s, s, 1 ], [1, s, s, 1 ], 'VALID')

class model_wrap(object):
    '''
    this is just for the easiness of model for pgd attacks
    '''
    def __init__(self,xent,pre_softmax,x_input,y_input,is_training, keep_prob):
        self.xent = xent
        self.x_input = x_input
        self.y_input = y_input
        self.pre_softmax = pre_softmax
        self.is_training = is_training
        self.keep_prob = keep_prob

def run_model(data, image_dim, label_count, depth,var_scope = 'densenet40'):
  train_steps = 234375
  weight_decay = 1e-4
  layers = int((depth - 4) / 3)
  graph = tf.Graph()
  old_vars = set(tf.global_variables())
  with graph.as_default():
    xs = tf.placeholder(tf.float32, shape=[None, image_dim, image_dim, 3])
    # now each image is standardized
    # xs_standardized = tf.map_fn(lambda img: tf.image.per_image_standardization(img),xs)
    ys = tf.placeholder(tf.float32, shape=[None, label_count])
    lr = tf.placeholder(tf.float32, shape=[])
    keep_prob = tf.placeholder(tf.float32)
    is_training = tf.placeholder(tf.bool, shape=[])

    # current = tf.reshape(xs, [ -1, 32, 32, 3 ])
    current = conv2d(xs, 3, 16, 3)

    current, features = block(current, layers, 16, 12, is_training, keep_prob)
    current = batch_activ_conv(current, features, features, 1, is_training, keep_prob)
    current = avg_pool(current, 2)
    current, features = block(current, layers, features, 12, is_training, keep_prob)
    current = batch_activ_conv(current, features, features, 1, is_training, keep_prob)
    current = avg_pool(current, 2)
    current, features = block(current, layers, features, 12, is_training, keep_prob)

    current = tf.contrib.layers.batch_norm(current, scale=True, is_training=is_training, updates_collections=None)
    current = tf.nn.relu(current)
    current = avg_pool(current, 8)
    final_dim = features
    current = tf.reshape(current, [ -1, final_dim ])
    Wfc = weight_variable([ final_dim, label_count ])
    bfc = bias_variable([ label_count ])
    logits = tf.matmul(current, Wfc) + bfc
    ys_ = tf.nn.softmax( logits )

    cross_entropy = -tf.reduce_mean(ys * tf.log(ys_ + 1e-12))
    l2 = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
    train_step = tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True).minimize(cross_entropy + l2 * weight_decay)
    correct_prediction = tf.equal(tf.argmax(ys_, 1), tf.argmax(ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    # define the attack graph, first wrap into "model" type object
    model = model_wrap(cross_entropy,logits,xs,ys,is_training, keep_prob)
    new_vars = set(tf.global_variables()) # track the number of newly created model variables
    attack = LinfPGDAttack(model,
                epsilon = 8.0,
                num_steps = 100,
                step_size = 2.0,
                random_start = True,
                loss_func = "xent")    

    saver = tf.train.Saver(var_list = new_vars - old_vars)
  model_file = tf.train.latest_checkpoint('model')
  print(model_file)
  if model_file is None:
    print('No model found')
    sys.exit()
  with tf.Session(graph=graph) as session:
    batch_size = 500
    saver.restore(session, model_file)

    # data set proper loading
    # train_data, train_labels = data['train_data'], data['train_labels']
    test_data, test_labels = data.eval_data.xs, data.eval_data.ys
    test_labels = keras.utils.to_categorical(test_labels, label_count)
    train_data, train_labels = data.train_data.xs, data.train_data.ys
    print('x_train shape:', train_data.shape)
    print('y_train shape:', train_labels.shape)
    print('x_test shape:', test_data.shape)
    print('y_test shape:', test_labels.shape)
    print(train_data.shape[0], 'train samples')
    print(test_data.shape[0], 'test samples')

    print("max pixel value:",np.max(train_data),np.max(test_data))
    print("min pixel value",np.min(train_data),np.min(test_data))
    # cifar = cifar10_input.AugmentedCIFAR10Data(data, session, model = 0)

    batch_count = int(len(test_data) / batch_size)
    # batches_data = np.split(train_data[:batch_count * batch_size], batch_count)
    # batches_labels = np.split(train_labels[:batch_count * batch_size], batch_count)
    print("Batch per epoch: ", batch_count)

    # check the natural accuracy
    nat_total_corr = 0
    for ibatch in range(batch_count):
      bstart = ibatch * batch_size
      bend = min(bstart + batch_size, len(test_data))
      print('batch size: {}'.format(bend - bstart))

      x_batch = test_data[bstart:bend, :]
      y_batch = test_labels[bstart:bend,:]
      corr_pred = session.run(correct_prediction,feed_dict = {xs:x_batch,ys:y_batch,is_training:False,keep_prob:1})
      nat_total_corr += np.sum(corr_pred)
    print("natural test accuracy:",nat_total_corr/len(test_data))
    # check the robust accuracy
    adv_total_corr = 0
    batch_size = 100
    batch_count = int(len(test_data) / batch_size)
    for ibatch in range(batch_count):
      bstart = ibatch * batch_size
      bend = min(bstart + batch_size, len(test_data))
      print('batch size: {}'.format(bend - bstart))

      x_batch = test_data[bstart:bend, :]
      y_batch = test_labels[bstart:bend,:]
      x_batch_adv = attack.perturb(x_batch, y_batch, session)
      corr_pred = session.run(correct_prediction,feed_dict = {xs:x_batch_adv,ys:y_batch,is_training:False,keep_prob:1})
      adv_total_corr += np.sum(corr_pred)
    print("robust test accuracy:",adv_total_corr/len(test_data))

class wrap_as_densenet_model(object):
  def __init__(self, depth,xs,ys,is_training,keep_prob,label_count):
    '''
    :param: xs: placeholder for input data, should be of shape (None,img_size,img_size,nchannel) 
    :param: ys: placeholder for input lable, should be of shape (None,class_num)
    :param: is_training: placeholder for determining if is in training phase, always ''False'' here
    :param: keep_prob: placeholder for dropout learning: always ''1'' here
    '''
    layers = int((depth - 4) / 3)
    # graph = tf.Graph()
    # with graph.as_default():
    # current = tf.reshape(xs, [ -1, 32, 32, 3 ])
    current = conv2d(xs, 3, 16, 3)
    current, features = block(current, layers, 16, 12, is_training, keep_prob)
    current = batch_activ_conv(current, features, features, 1, is_training, keep_prob)
    current = avg_pool(current, 2)
    current, features = block(current, layers, features, 12, is_training, keep_prob)
    current = batch_activ_conv(current, features, features, 1, is_training, keep_prob)
    current = avg_pool(current, 2)
    current, features = block(current, layers, features, 12, is_training, keep_prob)

    current = tf.contrib.layers.batch_norm(current, scale=True, is_training=is_training, updates_collections=None)
    current = tf.nn.relu(current)
    current = avg_pool(current, 8)
    final_dim = features
    current = tf.reshape(current, [ -1, final_dim ])
    Wfc = weight_variable([ final_dim, label_count ])
    bfc = bias_variable([ label_count ])
    logits = tf.matmul(current, Wfc) + bfc
    self.pre_softmax = logits
    ys_ = tf.nn.softmax( logits )
    self.softmax_prob = ys_
    mean_cross_entropy = -tf.reduce_mean(ys * tf.log(ys_ + 1e-12))
    self.xent = tf.nn.softmax_cross_entropy_with_logits(labels = ys,logits = logits)
    correct_prediction = tf.equal(tf.argmax(ys_, 1), tf.argmax(ys, 1))
    self.correct_prediction = correct_prediction
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    self.accuracy = accuracy

def run():
  # data_dir = 'data'
  data_path = '/zf18/fs5xz/Research_Code/tf_cifar10_train_code/vgg-tensorflow-cifar10/cifar10_data/'
  image_size = 32
  labels = 10
  # image_dim = image_size * image_size * 3
  # meta = unpickle(data_dir + '/batches.meta')
  # label_names = meta['label_names']
  # label_count = len(label_names)

  # train_files = [ 'data_batch_%d' % d for d in xrange(1, 6) ]
  # train_data, train_labels = load_data(train_files, data_dir, label_count)
  # pi = np.random.permutation(len(train_data))
  # train_data, train_labels = train_data[pi], train_labels[pi]
  # test_data, test_labels = load_data([ 'test_batch' ], data_dir, label_count)
  # print "Train:", np.shape(train_data), np.shape(train_labels)
  # print "Test:", np.shape(test_data), np.shape(test_labels)
  # data = { 'train_data': train_data,
  #     'train_labels': train_labels,
  #     'test_data': test_data,
  #     'test_labels': test_labels }

  raw_cifar = cifar10_input.CIFAR10Data(data_path)
  run_model(raw_cifar, image_size, labels, 40)

# run()
