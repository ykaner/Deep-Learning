import math
from functools import reduce

import tensorflow as tf


def variable_summaries(var):
	with tf.device('/cpu:0'):
		"""Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
		with tf.variable_scope('summaries'):
			mean = tf.reduce_mean(var)
			tf.summary.scalar('mean', mean)
			with tf.variable_scope('stddev'):
				stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
			tf.summary.scalar('stddev', stddev)
			tf.summary.scalar('max', tf.reduce_max(var))
			tf.summary.scalar('min', tf.reduce_min(var))
			tf.summary.histogram('histogram', var)


def residual_block(input_tensor, ksize, shapes, dropout=None, layer_name='res_block', option='A', active_prob=1.0):
	shape_in, shape_out = shapes
	shapes = int(shape_in), int(shape_out)
	
	if dropout is not None:
		dropout = 1
	
	reshape = shape_in != shape_out
	
	with tf.variable_scope(layer_name):
		shortcut_1 = input_tensor if not reshape else shortcut(input_tensor, shapes, layer_name='shortcut', option=option)
		
		def conv_act(out, name):
			random_tensor = active_prob
			random_tensor += tf.random_uniform(out.shape[0], dtype=out.dtype)
			is_active = tf.floor(random_tensor)
			
			out_shape = out.shape
			out = tf.transpose(tf.reshape(out, [-1, reduce((lambda _a, _b: _a * _b), out_shape[1:])]), [0, 1])
			out = out * is_active
			out = tf.reshape(tf.transpose(out, [0, 1]), out_shape)
			
			# result = tf.nn.relu(out * is_active + shortcut_1, name)
			# result = tf.cond(tf.equal(is_active, 1), lambda: shortcut_1, lambda: out + shortcut_1)
			# result = tf.nn.relu(result, name)
			return tf.nn.relu(out + shortcut_1, name)
		
		strides = [1] * 4 if not reshape else [1, 2, 2, 1]
		conv1 = conv2d_layer(input_tensor, [ksize, ksize, shape_in, shape_out], layer_name="conv_1", batch_n=True,
		                     strides=strides, act=tf.nn.relu)
		
		drop_layer = tf.nn.dropout(conv1, dropout)
		
		conv2 = conv2d_layer(drop_layer, [ksize, ksize, shape_out, shape_out], layer_name="conv_2", batch_n=True,
		                     act=conv_act)
		
	
	return conv2


def conv2d_layer(input_tensor, weights_shape, layer_name, strides=None, padding="SAME", batch_n=False, is_train=True,
                 act=tf.nn.relu):
	if strides is None:
		strides = [1, 1, 1, 1]
	if act is None:
		def no_act(val, name=''):
			return val
		
		act = no_act
	with tf.variable_scope(layer_name):
		# This Variable will hold the state of the weights for the layer
		with tf.variable_scope('weights'):
			weights = weight_variable(weights_shape)
			variable_summaries(weights)
		with tf.variable_scope('biases'):
			biases = bias_variable([weights_shape[-1]])
			variable_summaries(biases)
		with tf.variable_scope('conv_plus_b'):
			preactivate = tf.nn.conv2d(input_tensor, weights, strides=strides, padding=padding,
			                           name="pre_activations") + biases
			with tf.device('/cpu:0'):
				tf.summary.histogram('pre_activations', preactivate)
		if batch_n:
			with tf.variable_scope('batch_normalization'):
				mean, var = tf.nn.moments(preactivate, [0], name='maen_var')
				z = tf.div(preactivate - mean, tf.sqrt(var + 1e-4))
				betta = tf.get_variable(name='betta', shape=weights_shape[-1], initializer=tf.zeros_initializer())
				gamma = tf.get_variable(name='gamma', shape=weights_shape[-1], initializer=tf.ones_initializer())
				preactivate = tf.add(tf.multiply(z, gamma), betta)
		# preactivate = tf.nn.batch_normalization(preactivate, mean, var, betta, gamma, 1e-3, name='batch_norm')
		activations = act(preactivate, name='activation')
		with tf.device('/cpu:0'):
			tf.summary.histogram('activations', activations)
		return activations


def shortcut(input_tensor, shapes, layer_name='shourtcut', option='A'):
	option = option.upper()
	
	in_shape, out_shape = shapes
	
	with tf.variable_scope(layer_name):
		pad = (out_shape - in_shape) / 2
		
		if option == 'A':
			x = avg_pool_layer(input_tensor, [1, 2, 2, 1], [1, 2, 2, 1], layer_name='shortcut_pool', padding='SAME')
			
			pads = [[0, 0]] * 3 + [[math.ceil(pad), math.floor(pad)]]
			x = tf.pad(x, paddings=pads)
		
		elif option == 'B':
			x = avg_pool_layer(input_tensor, [1, 2, 2, 1], [1, 2, 2, 1], layer_name='shortcut_pool', padding='SAME')

			W_s = weight_variable(shapes)
			x = tf.matmul(x, W_s)
		
		elif option == 'C':
			x = conv2d_layer(input_tensor, [1, 1, in_shape, out_shape], layer_name='shortcut_conv',
			                 strides=[1, 2, 2, 1], batch_n=False, act=None)
		
		else:
			x = input_tensor
	return x


def avg_pool_layer(input_tensor, ksize, strides, layer_name, padding='SAME'):
	with tf.variable_scope(layer_name):
		return tf.nn.avg_pool(input_tensor, ksize=ksize, strides=strides, padding=padding, name='avg_pool')


def pool_layer(input_tensor, ksize, strides, layer_name, padding="SAME"):
	with tf.variable_scope(layer_name):
		return tf.nn.max_pool_with_argmax(input_tensor, ksize=ksize, strides=strides, padding=padding, name='max_pool')


# We can't initialize these variables to 0 - the network will get stuck.
def weight_variable(shape, stddev=0.1, name="weights"):
	"""Create a weight variable with appropriate initialization."""
	reg_betta = 0.0001
	regularizer = tf.contrib.layers.l2_regularizer(reg_betta)
	return tf.get_variable(name=name, shape=shape, initializer=tf.truncated_normal_initializer(stddev=stddev),
	                       regularizer=regularizer)


def bias_variable(shape, name='variable'):
	"""Create a bias variable with appropriate initialization."""
	return tf.get_variable(name=name, shape=shape, initializer=tf.truncated_normal_initializer(stddev=0.1))

# def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
# 	"""Reusable code for making a simple neural net layer.
#
# 	It does a matrix multiply, bias add, and then uses relu to nonlinearize.
# 	It also sets up name scoping so that the resultant graph is easy to read,
# 	and adds a number of summary ops.
# 	"""
# 	# Adding a name scope ensures logical grouping of the layers in the graph.
# 	with tf.variable_scope(layer_name):
# 		# This Variable will hold the state of the weights for the layer
# 		with tf.variable_scope('weights'):
# 			weights = weight_variable([input_dim, output_dim])
# 			variable_summaries(weights)
# 		with tf.variable_scope('biases'):
# 			biases = bias_variable([output_dim])
# 			variable_summaries(biases)
# 		with tf.variable_scope('Wx_plus_b'):
# 			preactivate = tf.matmul(input_tensor, weights) + biases
# 			tf.summary.histogram('pre_activations', preactivate)
# 		activations = act(preactivate, name='activation')
# 		tf.summary.histogram('activations', activations)
# 		return activations


# hidden1 = nn_layer(x, 784, 500, 'layer1')

# with tf.variable_scope('dropout'):
# 	keep_prob = tf.placeholder(tf.float32)
# 	tf.summary.scalar('dropout_keep_probability', keep_prob)
# 	dropped = tf.nn.dropout(hidden1, keep_prob)

# # Do not apply softmax activation yet, see below.
# y = nn_layer(dropped, 500, 10, 'layer2', act=tf.identity)

# with tf.variable_scope('cross_entropy'):
# 	The raw formulation of cross-entropy,
#
# 	tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.softmax(y)),
# 	                              reduction_indices=[1]))
#
# 	can be numerically unstable.
#
# 	So here we use tf.losses.sparse_softmax_cross_entropy on the
# 	raw logit outputs of the nn_layer above.
# 	with tf.variable_scope('total'):
# 		cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=y_, logits=y)

# tf.summary.scalar('cross_entropy', cross_entropy)

# with tf.variable_scope('train'):
# 	train_step = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(
# 			cross_entropy)
#
# with tf.variable_scope('accuracy'):
# 	with tf.variable_scope('correct_prediction'):
# 		correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
# 	with tf.variable_scope('accuracy'):
# 		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# tf.summary.scalar('accuracy', accuracy)

# Merge all the summaries and write them out to /tmp/mnist_logs (by default)
# merged = tf.summary.merge_all()
# train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train',
#                                      sess.graph)
# test_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/test')
# tf.global_variables_initializer().run()
