from __future__ import absolute_import, division, print_function

import math
import os.path
import re
import time
from datetime import datetime

import numpy as np
import tensorflow as tf
from six.moves import range  # pylint: disable=redefined-builtin

from . import cifar10
from . import cifar10_eval

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('train_dir', '/tmp/cifar10_train', """Directory where to write event logs and checkpoint.""")
tf.flags.DEFINE_integer('max_epochs', 150, """Number of batches to run.""")
tf.flags.DEFINE_integer('num_gpus', 4, """How many GPUs to use.""")
tf.flags.DEFINE_boolean('log_device_placement', False, """Whether to log device placement.""")
tf.flags.DEFINE_integer('max_steps', 100000, 'Dont use it. calculated out from max_epochs')
tf.flags.DEFINE_boolean('is_eval', False, 'is evaluate each epoch')

FLAGS.max_steps = math.ceil(FLAGS.max_epochs * cifar10.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size)


def tower_loss(scope, images, labels, keep_prob=1.0, last_active_prob=1.0):
	"""
	Calculate the total loss on a single tower running the CIFAR model.
	
	:arg scope: unique prefix string identifying the CIFAR tower, e.g. 'tower_0'
	:arg images: Images. 4D tensor of shape [batch_size, height, width, 3].
	:arg labels: Labels. 1D tensor of shape [batch_size].
	
	:returns: Tensor of shape [] containing the total loss for a batch of data
	"""
	
	# Build ResNet Graph.
	logits = cifar10.ResNet(images, keep_prob=keep_prob, last_active_prob=last_active_prob)
	
	with tf.name_scope('total'):
		y_pred_cls = tf.argmax(logits, axis=1, name="y_pred_cls")
		
		correct_prediction = tf.equal(y_pred_cls, tf.cast(labels, tf.int64), name="correct_predictions")
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="accuracy")
		
		tf.add_to_collection('accuracy', accuracy)
	
	# Build the portion of the Graph calculating the losses. Note that we will
	# assemble the total_loss using a custom function below.
	_ = cifar10.loss(logits, labels)
	
	# Assemble all of the losses for the current tower only.
	losses = tf.get_collection('losses', scope)
	
	# Calculate the total loss for the current tower.
	total_loss = tf.add_n(losses, name='total_loss')
	
	# Attach a scalar summary to all individual losses and the total loss; do the
	# same for the averaged version of the losses.
	for l in losses + [total_loss]:
		# Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
		# session. This helps the clarity of presentation on tensorboard.
		loss_name = re.sub('{0}_[0-9]*/'.format(cifar10.TOWER_NAME), '', l.op.name)
		tf.summary.scalar(loss_name, l)
	
	accuracies = tf.get_collection('accuracy')
	
	for ac in accuracies:
		ac_name = re.sub('{0}_[0-9]*/'.format(cifar10.TOWER_NAME), '', ac.op.name)
		tf.summary.scalar(ac_name, ac)
	
	return total_loss, accuracy


def average_gradients(tower_grads):
	"""
	Calculate the average gradient for each shared variable across all towers.
	
	Note that this function provides a synchronization point across all towers.
	
	:arg tower_grads: List of lists of (gradient, variable) tuples. The outer list
		is over individual gradients. The inner list is over the gradient
		calculation for each tower.
	
	:returns:
		List of pairs of (gradient, variable) where the gradient has been averaged across all towers.
	"""
	average_grads = []
	for grad_and_vars in zip(*tower_grads):
		# Note that each grad_and_vars looks like the following:
		#   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
		grads = []
		for g, _ in grad_and_vars:
			# Add 0 dimension to the gradients to represent the tower.
			expanded_g = tf.expand_dims(g, 0)
			
			# Append on a 'tower' dimension which we will average over below.
			grads.append(expanded_g)
		
		# Average over the 'tower' dimension.
		grad = tf.concat(axis=0, values=grads)
		grad = tf.reduce_mean(grad, 0)
		
		# Keep in mind that the Variables are redundant because they are shared
		# across towers. So .. we will just return the first tower's pointer to
		# the Variable.
		v = grad_and_vars[0][1]
		grad_and_var = (grad, v)
		average_grads.append(grad_and_var)
	return average_grads


def train():
	"""Train CIFAR-10 for a number of steps."""
	with tf.Graph().as_default() as g, tf.device('/cpu:0'):
		# Create a variable to count the number of train() calls. This equals the
		# number of batches processed * FLAGS.num_gpus.
		global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
		
		# Calculate the learning rate schedule.
		num_batches_per_epoch = (cifar10.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN // FLAGS.batch_size)
		decay_steps = int(num_batches_per_epoch * cifar10.NUM_EPOCHS_PER_DECAY)
		
		# Decay the learning rate exponentially based on the number of steps.
		# lr = tf.train.exponential_decay(
		# 		cifar10.INITIAL_LEARNING_RATE, global_step, decay_steps, cifar10.LEARNING_RATE_DECAY_FACTOR,
		# 		staircase=True)
		
		keep_prob = tf.placeholder(tf.float32, name='keep_prob')
		last_active_prob = tf.placeholder(tf.float32, name='last_active_prob')
		
		lr = tf.placeholder(tf.float32, name='learning_rate')
		
		# Create an optimizer that performs gradient descent.
		# opt = tf.train.GradientDescentOptimizer(lr)
		# opt = tf.train.MomentumOptimizer(lr, momentum=0.9, use_nesterov=True, name='Momentum_Optimizer')
		opt = tf.train.AdamOptimizer(lr)  # just for the second plot
		
		# Get images and labels for CIFAR-10.
		images, labels = cifar10.distorted_inputs()
		batch_queue = tf.contrib.slim.prefetch_queue.prefetch_queue([images, labels], capacity=2 * FLAGS.num_gpus)
		# Calculate the gradients for each model tower.
		tower_grads = []
		with tf.variable_scope(tf.get_variable_scope()):
			for i in range(FLAGS.num_gpus):
				with tf.device('/gpu:{:d}'.format(i)):
					with tf.name_scope('{}_{:d}'.format(cifar10.TOWER_NAME, i)) as scope:
						# Dequeues one batch for the GPU
						image_batch, label_batch = batch_queue.dequeue()
						# Calculate the loss for one tower of the CIFAR model. This function
						# constructs the entire CIFAR model but shares the variables across
						# all towers.
						loss, acc = tower_loss(scope, image_batch, label_batch, keep_prob=keep_prob,
						                       last_active_prob=last_active_prob)
						
						# Reuse variables for the next tower.
						tf.get_variable_scope().reuse_variables()
						
						# Retain the summaries from the final tower.
						summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)
						
						# Calculate the gradients for the batch of data on this CIFAR tower.
						grads = opt.compute_gradients(loss)
						
						# Keep track of the gradients across all towers.
						tower_grads.append(grads)
		
		# We must calculate the mean of each gradient. Note that this is the
		# synchronization point across all towers.
		grads = average_gradients(tower_grads)
		
		# Add a summary to track the learning rate.
		summaries.append(tf.summary.scalar('learning_rate', lr))
		
		# Add histograms for gradients.
		for grad, var in grads:
			if grad is not None:
				summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))
		
		# Apply the gradients to adjust the shared variables.
		apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
		
		# Add histograms for trainable variables.
		for var in tf.trainable_variables():
			summaries.append(tf.summary.histogram(var.op.name, var))
		
		# Track the moving averages of all trainable variables.
		variable_averages = tf.train.ExponentialMovingAverage(cifar10.MOVING_AVERAGE_DECAY, global_step)
		variables_averages_op = variable_averages.apply(tf.trainable_variables())
		
		# Group all updates to into a single train op.
		train_op = tf.group(apply_gradient_op, variables_averages_op)
		
		# Create a saver.
		saver = tf.train.Saver(tf.global_variables())
		
		# Build the summary operation from the last tower summaries.
		summary_op = tf.summary.merge(summaries)
		
		# Build an initialization operation to run below.
		init = tf.global_variables_initializer()
		
		# Start running operations on the Graph. allow_soft_placement must be set to
		# True to build towers on GPU, as some of the ops do not have GPU
		# implementations.
		
		config = tf.ConfigProto(
				allow_soft_placement=True,
				log_device_placement=FLAGS.log_device_placement)
		config.gpu_options.allow_growth = True
		sess = tf.Session(config=config)
		sess.run(init)
		
		# Start the queue runners.
		tf.train.start_queue_runners(sess=sess)
		
		summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)
		
		def get_total_parameters():
			"""
			Calculate how match parameters do we have in the graph.
			:return: The number of parameters in the graph.
			"""
			total_parameters = 0
			for variable in tf.trainable_variables():
				shape = variable.get_shape()
				variable_parameters = 1
				for dim in shape:
					variable_parameters *= dim.value
				total_parameters += variable_parameters
			print(total_parameters)
			dtime = datetime.now()
			dtime = str(dtime.replace(microsecond=0, second=0, minute=0))
			with open('total_parameters' + dtime + '.txt', 'w') as f:
				f.write(str(total_parameters))
		
		get_total_parameters()
		
		def lr_dict(step):
			epoch = step // num_batches_per_epoch
			if epoch < 60:
				return 0.1
			elif epoch < 90:
				return 0.01
			else:
				return 0.001
		
		epoch = 0
		epoch_time = time.time()
		for step in range(FLAGS.max_steps):
			start_time = time.time()
			_, loss_value, acc_value = sess.run([train_op, loss, acc],
			                                    feed_dict={lr: lr_dict(step), keep_prob: 0.5, last_active_prob: 0.5})
			duration = time.time() - start_time
			
			assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
			
			if step % 10 == 0:
				num_examples_per_step = FLAGS.batch_size * FLAGS.num_gpus
				examples_per_sec = num_examples_per_step / duration
				sec_per_batch = duration / FLAGS.num_gpus
				
				print(
						'\r{timestamp}: Epoch-{epoch:03d}, step {step:d}, loss = {loss:.2f}, acc = {acc:f} ({example_rate:.1f} examples/sec; {batch_rate:.3f} sec/batch)'.format(
								timestamp=datetime.now(),
								epoch=epoch,
								step=step % num_batches_per_epoch,
								loss=loss_value,
								acc=acc_value,
								example_rate=examples_per_sec,
								batch_rate=sec_per_batch),
						end='')
			
			new_epoch = step // num_batches_per_epoch
			# evaluate if new epoch started
			global acc_file
			acc_file = 'accuracy_' + str(datetime.now().replace(microsecond=0, second=0,
			                                                    minute=0)) + '.txt' if 'acc_file' not in globals() else acc_file
			if new_epoch > epoch:
				epoch = new_epoch
				new_epoch_time = time.time()
				print('\nthis epoch took: ' + str(new_epoch_time - epoch_time) + ' time')
				epoch_time = new_epoch_time
				
				with open(acc_file, 'a' if os.path.exists(acc_file) else 'w') as f:
					f.write(str(acc_value) + '\n')
				
				# save checkpoint
				checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
				saver.save(sess, checkpoint_path, global_step=step)
				
				if FLAGS.is_eval:
					ev_time = time.time()
					cifar10_eval.main()
					print('eval time:' + str(time.time() - ev_time))
			
			if step % 100 == 0:
				summary_str = sess.run(summary_op, feed_dict={lr: lr_dict(step), keep_prob: 0.5, last_active_prob: 0.5})
				summary_writer.add_summary(summary_str, step)


def main(argv=None):  # pylint: disable=unused-argument
	cifar10.maybe_download_and_extract()
	if tf.gfile.Exists(FLAGS.train_dir):
		tf.gfile.DeleteRecursively(FLAGS.train_dir)
	tf.gfile.MakeDirs(FLAGS.train_dir)
	train()


if __name__ == '__main__':
	tf.app.run()