# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from __future__ import absolute_import, division, print_function

__doc__ = """Builds the CIFAR-10 network.

Summary of available functions:

 # Compute input images and labels for training. If you would like to run
 # evaluations, use inputs() instead.
 inputs, labels = distorted_inputs()

 # Compute inference on the model inputs to make a prediction.
 predictions = inference(inputs)

 # Compute the total loss of the prediction with respect to the labels.
 loss = loss(predictions, labels)

 # Create a graph to run one step of training with respect to the loss.
 train_op = train(loss, global_step)
"""

import os
import re
import sys
import tarfile

from . import cifar10_input
from utils import *
import tensorflow as tf
from six.moves import urllib

FLAGS = tf.flags.FLAGS

# Basic model parameters.
tf.flags.DEFINE_integer('batch_size', 128,
                        """Number of images to process in a batch.""")
tf.flags.DEFINE_string('data_dir', '/tmp/cifar10_data',
                       """Path to the CIFAR-10 data directory.""")
tf.flags.DEFINE_boolean('use_fp16', False,
                        """Train the model using fp16.""")

# Global constants describing the CIFAR-10 data set.
IMAGE_SIZE = cifar10_input.IMAGE_SIZE
NUM_CLASSES = cifar10_input.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999  # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0  # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1  # Initial learning rate.

# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'

DATA_URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'


def _activation_summary(x):
	"""
	Helper to create summaries for activations.
	
	Creates a summary that provides a histogram of activations.
	Creates a summary that measures the sparsity of activations.
	
	:arg x: Tensor
	:returns: nothing
	"""
	# Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
	# session. This helps the clarity of presentation on functions.
	tensor_name = re.sub('{0}_[0-9]*/'.format(TOWER_NAME), '', x.op.name)
	tf.summary.histogram(tensor_name + '/activations', x)
	tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def _variable_on_cpu(name, shape, initializer):
	"""
	Helper to create a Variable stored on CPU memory.
	
	:arg name: name of the variable
	:arg shape: list of ints
	:arg initializer: initializer for Variable
	:returns: Variable Tensor
	"""
	with tf.device('/cpu:0'):
		dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
		var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
	return var


def _variable_with_weight_decay(name, shape, stddev, wd):
	"""
	Helper to create an initialized Variable with weight decay.
	
	Note that the Variable is initialized with a truncated normal distribution.
	A weight decay is added only if one is specified.
	
	:arg name: name of the variable
	:arg shape: list of ints
	:arg stddev: standard deviation of a truncated Gaussian
	:arg wd: add L2Loss weight decay multiplied by this float. If None, weight decay is not added for this Variable.
	
	:returns: Variable Tensor
	"""
	dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
	var = _variable_on_cpu(
			name,
			shape,
			tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
	if wd is not None:
		weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
		tf.add_to_collection('losses', weight_decay)
	return var


def distorted_inputs():
	"""
	Construct distorted input for CIFAR training using the Reader ops.
	
	:returns images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
	:returns labels: Labels. 1D tensor of [batch_size] size.
	
	:raises ValueError: If no data_dir
	"""
	if not FLAGS.data_dir:
		raise ValueError('Please supply a data_dir')
	data_dir = os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin')
	images, labels = cifar10_input.distorted_inputs(data_dir=data_dir, batch_size=FLAGS.batch_size)
	if FLAGS.use_fp16:
		images = tf.cast(images, tf.float16)
		labels = tf.cast(labels, tf.float16)
	return images, labels


def inputs(eval_data):
	"""
	Construct input for CIFAR evaluation using the Reader ops.
	
	:arg eval_data: bool, indicating if one should use the train or eval data set.
	
	:returns images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
	:returns labels: Labels. 1D tensor of [batch_size] size.
	
	:raises: ValueError: If no data_dir
	"""
	if not FLAGS.data_dir:
		raise ValueError('Please supply a data_dir')
	data_dir = os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin')
	images, labels = cifar10_input.inputs(eval_data=eval_data, data_dir=data_dir, batch_size=FLAGS.batch_size)
	if FLAGS.use_fp16:
		images = tf.cast(images, tf.float16)
		labels = tf.cast(labels, tf.float16)
	return images, labels


def inference(images):
	_NUM_CLASSES = 10
	keep_prob = 1.0
	net_option = 'A'
	with tf.variable_scope('ResNet'):
		conv0 = functions.conv2d_layer(images, [3, 3, 3, 16], layer_name="conv_0", batch_n=False)
		
		def conv1_act(out, name):
			return tf.nn.relu(out + conv0, name)
		
		conv1 = functions.conv2d_layer(conv0, [3, 3, 16, 16], layer_name="conv_1", batch_n=True,
		                               act=tf.nn.relu)
		
		conv1_1 = functions.conv2d_layer(conv1, [3, 3, 16, 16], layer_name="conv_1_1", batch_n=True,
		                                 act=conv1_act)
		
		shortcut1_1 = conv1_1
		
		def conv1_2_act(out, name):
			return tf.nn.relu(out + shortcut1_1, name)
		
		conv1_2 = functions.conv2d_layer(conv1_1, [3, 3, 16, 16], layer_name="conv_1_2", batch_n=True,
		                                 act=tf.nn.relu)
		
		conv1_3 = functions.conv2d_layer(conv1_2, [3, 3, 16, 16], layer_name="conv_1_3", batch_n=True,
		                                 act=conv1_2_act)
		
		block1 = conv1_3
		
		shortcut2 = functions.shortcut(block1, [16, 32], layer_name='shortcut2', option=net_option)
		
		def conv2_act(out, name):
			return tf.nn.relu(out + shortcut2, name)
		
		conv2 = functions.conv2d_layer(block1, [3, 3, 16, 32], layer_name="conv_2", strides=[1, 2, 2, 1],
		                               batch_n=True, act=tf.nn.relu)
		
		conv2_1 = functions.conv2d_layer(conv2, [3, 3, 32, 32], layer_name="conv_2_1", batch_n=True,
		                                 act=conv2_act)
		
		shortcut2_2 = conv2_1
		
		def conv2_2_act(out, name):
			return tf.nn.relu(out + shortcut2_2, name)
		
		conv2_2 = functions.conv2d_layer(conv2_1, [3, 3, 32, 32], layer_name="conv_2_2", batch_n=True,
		                                 act=tf.nn.relu)
		
		conv2_3 = functions.conv2d_layer(conv2_2, [3, 3, 32, 32], layer_name="conv_2_3", batch_n=True,
		                                 act=conv2_2_act)
		
		block2 = conv2_3
		
		shortcut3 = functions.shortcut(block2, [32, 64], layer_name='shortcut3', option=net_option)
		
		def conv3_act(out, name):
			return tf.nn.relu(out + shortcut3, name)
		
		conv3 = functions.conv2d_layer(block2, [3, 3, 32, 64], layer_name="conv_3", strides=[1, 2, 2, 1],
		                               batch_n=True, act=tf.nn.relu)
		
		conv3_1 = functions.conv2d_layer(conv3, [3, 3, 64, 64], layer_name="conv_3_1", batch_n=True,
		                                 act=conv3_act)
		
		shortcut3_2 = conv3_1
		
		def conv3_2_act(out, name):
			return tf.nn.relu(out + shortcut3_2, name)
		
		conv3_2 = functions.conv2d_layer(conv3_1, [3, 3, 64, 64], layer_name="conv_3_2", batch_n=True,
		                                 act=tf.nn.relu)
		
		conv3_3 = functions.conv2d_layer(conv3_2, [3, 3, 64, 64], layer_name="conv_3_3", batch_n=True,
		                                 act=conv3_2_act)
		
		def conv3_3_act(out, name):
			return tf.nn.relu(out + conv3_3, name)
		
		conv3_4 = functions.conv2d_layer(conv3_3, [1, 1, 64, 64], layer_name="conv_3_4", batch_n=True,
		                                 act=tf.nn.relu)
		
		conv3_5 = functions.conv2d_layer(conv3_4, [1, 1, 64, 64], layer_name="conv_3_5", batch_n=True,
		                                 act=conv3_3_act)
		
		gap = tf.layers.average_pooling2d(conv3_5, [6, 6], [6, 6], padding='VALID', name='gap')
		
		flat = tf.reshape(gap, [-1, 64], name="flat")
		
		with tf.variable_scope('fc_1'):
			fc = tf.nn.relu(tf.layers.dense(inputs=flat, units=32, name="dense_layer"),
			                name="relu")  # , activation=tf.nn.relu)
			drop4 = tf.nn.dropout(fc, keep_prob, name="dropout")
		
		with tf.device('cpu:0'):
			tf.summary.histogram("drop4", drop4)
		
		logits = tf.nn.softmax(tf.layers.dense(inputs=drop4, units=_NUM_CLASSES), name="softmax")
	
	return logits


def loss(logits, labels):
	"""
	Add L2Loss to all the trainable variables.
	
	Add summary for "Loss" and "Loss/avg".
	
	:arg logits: Logits from inference().
	:arg labels: Labels from distorted_inputs or inputs(). 1-D tensor of shape [batch_size]
	
	:returns: Loss tensor of type float.
	"""
	# Calculate the average cross entropy loss across the batch.
	labels = tf.cast(labels, tf.int64)
	cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
			labels=labels, logits=logits, name='cross_entropy_per_example')
	cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
	tf.add_to_collection('losses', cross_entropy_mean)
	
	# The total loss is defined as the cross entropy loss plus all of the weight
	# decay terms (L2 loss).
	return tf.add_n(tf.get_collection('losses'), name='total_loss')


def _add_loss_summaries(total_loss):
	"""
	Add summaries for losses in CIFAR-10 model.
	
	Generates moving average for all losses and associated summaries for
	visualizing the performance of the network.
	
	:arg total_loss: Total loss from loss().
	:returns: loss_averages_op: op for generating moving averages of losses.
	"""
	# Compute the moving average of all individual losses and the total loss.
	loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
	losses = tf.get_collection('losses')
	loss_averages_op = loss_averages.apply(losses + [total_loss])
	
	# Attach a scalar summary to all individual losses and the total loss; do the
	# same for the averaged version of the losses.
	for l in losses + [total_loss]:
		# Name each loss as '(raw)' and name the moving average version of the loss
		# as the original loss name.
		tf.summary.scalar(l.op.name + ' (raw)', l)
		tf.summary.scalar(l.op.name, loss_averages.average(l))
	
	return loss_averages_op


def train(total_loss, global_step):
	"""
	Train CIFAR-10 model.
	
	Create an optimizer and apply to all trainable variables. Add moving
	average for all trainable variables.
	
	:arg total_loss: Total loss from loss().
	:arg global_step: Integer Variable counting the number of training steps processed.
	
	:returns: train_op: op for training.
	"""
	# Variables that affect learning rate.
	num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
	decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)
	
	# Decay the learning rate exponentially based on the number of steps.
	lr = tf.train.exponential_decay(
			INITIAL_LEARNING_RATE, global_step, decay_steps, LEARNING_RATE_DECAY_FACTOR, staircase=True)
	tf.summary.scalar('learning_rate', lr)
	
	# Generate moving averages of all losses and associated summaries.
	loss_averages_op = _add_loss_summaries(total_loss)
	
	# Compute gradients.
	with tf.control_dependencies([loss_averages_op]):
		opt = tf.train.GradientDescentOptimizer(lr)
		grads = opt.compute_gradients(total_loss)
	
	# Apply gradients.
	apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
	
	# Add histograms for trainable variables.
	for var in tf.trainable_variables():
		tf.summary.histogram(var.op.name, var)
	
	# Add histograms for gradients.
	for grad, var in grads:
		if grad is not None:
			tf.summary.histogram(var.op.name + '/gradients', grad)
	
	# Track the moving averages of all trainable variables.
	variable_averages = tf.train.ExponentialMovingAverage(
			MOVING_AVERAGE_DECAY, global_step)
	with tf.control_dependencies([apply_gradient_op]):
		variables_averages_op = variable_averages.apply(tf.trainable_variables())
	
	return variables_averages_op


def maybe_download_and_extract():
	"""
	Download and extract the tarball from Alex's website.
	"""
	dest_directory = FLAGS.data_dir
	if not os.path.exists(dest_directory):
		os.makedirs(dest_directory)
	filename = DATA_URL.split('/')[-1]
	filepath = os.path.join(dest_directory, filename)
	if not os.path.exists(filepath):
		def _progress(count, block_size, total_size):
			sys.stdout.write(
					'\r>> Downloading {} {:.1f}%'.format(
							filename, float(count * block_size) / float(total_size) * 100.0))
			sys.stdout.flush()
		
		filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
		print()
		statinfo = os.stat(filepath)
		print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
	extracted_dir_path = os.path.join(dest_directory, 'cifar-10-batches-bin')
	if not os.path.exists(extracted_dir_path):
		tarfile.open(filepath, 'r:gz').extractall(dest_directory)
