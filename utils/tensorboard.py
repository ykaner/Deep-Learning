import math
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
