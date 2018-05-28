#! python3
from __future__ import absolute_import

import glob
import math
import os
import pickle
import re
import sys
import tarfile
import zipfile
from datetime import datetime
from time import time
from urllib.request import urlretrieve

import numpy as np
import tensorflow as tf

import utils.tensorboard

if os.name is "nt":
	tmp_path = "C:/tmp/"
else:
	tmp_path = "/tmp/"

tensorboard_train_counter = 0
tensorboard_test_counter = 0

sess = tf.Session()


def pretty_time():
	return str(datetime.now().replace(microsecond=0)).replace(':', '-')


def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)


def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)


def max_pool_2x2(x, name):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)


def avg_pool_2x2(x, name):
	return tf.nn.avg_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)


def tower_loss(scope, images, labels):
	"""
	Calculate the total loss on a single tower running the CIFAR model.
	
	:param scope: unique prefix string identifying the CIFAR tower, e.g. 'tower_0'
	:param images: Images. 4D tensor of shape [batch_size, height, width, 3].
	:param labels: Labels. 1D tensor of shape [batch_size].
	:returns: Tensor of shape [] containing the total loss for a batch of data
	"""
	
	# Build inference Graph.
	logits = cifar10.inference(images)
	
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
		loss_name = re.sub('tower_[0-9]*/', '', l.op.name)
		tf.summary.scalar(loss_name, l)
	
	return total_loss


def average_gradients(tower_grads):
	"""
	Calculate the average gradient for each shared variable across all towers.
	
	Note that this function provides a synchronization point across all towers.
	
	:param tower_grads: List of lists of (gradient, variable) tuples. The outer list is over individual gradients. The inner list is over the gradient calculation for each tower.
	:returns: List of pairs of (gradient, variable) where the gradient has been averaged across all towers.
	"""
	
	average_grads = []
	for grad_and_vars in zip(*tower_grads):
		# Note that each grad_and_vars looks like the following:
		#   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
		grads = []
		
		print('grad_and_vars: ')
		print(grad_and_vars)
		cur_shape = []
		for g, _ in grad_and_vars:
			# Add 0 dimension to the gradients to represent the tower.
			print(g)
			if not cur_shape and g is not None:
				cur_shape = g.shape
			g = tf.zeros(cur_shape) if g is None else g
			expanded_g = tf.expand_dims(g, 0)
			# Append on a 'tower' dimension which we will average over below.
			grads.append(expanded_g)
		
		print(grads)
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


def ResNet(_x, keep_prob, reuse):
	with tf.variable_scope('ResNet', reuse=reuse):
		conv0 = utils.tensorboard.conv2d_layer(_x, [3, 3, 3, 16], layer_name="conv_0")
		
		def conv1_act(out, name):
			return tf.nn.relu(out + conv0, name)
		
		conv1 = utils.tensorboard.conv2d_layer(conv0, [3, 3, 16, 16], layer_name="conv_1", act=tf.nn.relu)
		
		conv1_1 = utils.tensorboard.conv2d_layer(conv1, [3, 3, 16, 16], layer_name="conv_1_1", act=conv1_act)
		
		# pool = max_pool_2x2(conv1_1, name="pool")
		
		drop = tf.nn.dropout(conv1_1, keep_prob, name="drop")
		
		def identical(val, name=''):
			return val
		
		shortcut2 = utils.tensorboard.conv2d_layer(drop, [1, 1, 16, 32], layer_name="shortcut2", strides=[1, 2, 2, 1],
		                                           act=identical)
		
		def conv2_act(out, name):
			return tf.nn.relu(out + shortcut2, name)
		
		conv2 = utils.tensorboard.conv2d_layer(drop, [3, 3, 16, 32], layer_name="conv_2", strides=[1, 2, 2, 1],
		                                       act=tf.nn.relu)
		
		conv2_1 = utils.tensorboard.conv2d_layer(conv2, [3, 3, 32, 32], layer_name="conv_2_1", act=conv2_act)
		
		shortcut2_2 = conv2_1
		
		def conv2_2_act(out, name):
			return tf.nn.relu(out + shortcut2_2, name)
		
		conv2_2 = utils.tensorboard.conv2d_layer(conv2_1, [3, 3, 32, 32], layer_name="conv_2_2", act=tf.nn.relu)
		
		conv2_3 = utils.tensorboard.conv2d_layer(conv2_2, [3, 3, 32, 32], layer_name="conv_2_3", act=conv2_2_act)
		
		pool2 = conv2_3  # max_pool_2x2(conv2_3, name="pool2")
		
		shortcut3 = utils.tensorboard.conv2d_layer(pool2, [1, 1, 32, 64], layer_name="shortcut3", strides=[1, 2, 2, 1],
		                                           act=identical)
		
		def conv3_act(out, name):
			return tf.nn.relu(out + shortcut3, name)
		
		conv3 = utils.tensorboard.conv2d_layer(pool2, [3, 3, 32, 64], layer_name="conv_3", strides=[1, 2, 2, 1],
		                                       act=tf.nn.relu)
		
		conv3_1 = utils.tensorboard.conv2d_layer(conv3, [3, 3, 64, 64], layer_name="conv_3_1", act=conv3_act)
		
		shortcut3_2 = conv3_1
		
		def conv3_2_act(out, name):
			return tf.nn.relu(out + shortcut3_2, name)
		
		conv3_2 = utils.tensorboard.conv2d_layer(conv3_1, [3, 3, 64, 64], layer_name="conv_3_2", act=tf.nn.relu)
		
		conv3_3 = utils.tensorboard.conv2d_layer(conv3_2, [3, 3, 64, 64], layer_name="conv_3_3", act=conv3_2_act)
		
		gap = tf.layers.average_pooling2d(conv3_3, [8, 8], [8, 8], padding='VALID', name='gap')
		
		# pool3 = avg_pool_2x2(conv3_3, name="pool3")
		
		flat = tf.reshape(gap, [-1, 64], name="flat")
		
		# with tf.variable_scope('fc_1'):
		# 	fc = tf.nn.relu(tf.layers.dense(inputs=flat, units=32, name="dense_layer"),
		# 	                name="relu")  # , activation=tf.nn.relu)
		# 	drop4 = tf.nn.dropout(fc, keep_prob, name="dropout")
		#
		# tf.summary.histogram("drop4", drop4)
		
		logits = tf.nn.softmax(tf.layers.dense(inputs=flat, units=_NUM_CLASSES), name="softmax")
	
	return logits


def model():
	_IMAGE_SIZE = 32
	_IMAGE_CHANNELS = 3
	with tf.device('/cpu:0'):
		tower_grads = []
		reuse_vars = False
		with tf.name_scope('input'):
			X = tf.placeholder(tf.float32, shape=[None, _IMAGE_SIZE * _IMAGE_SIZE * _IMAGE_CHANNELS], name='Input')
			Y = tf.placeholder(tf.float32, shape=[None, _NUM_CLASSES], name='Output')
		
		with tf.name_scope('input_reshape'):
			X_image = tf.reshape(X, [-1, _IMAGE_SIZE, _IMAGE_SIZE, _IMAGE_CHANNELS], name='images')
		
		tf.summary.image("intput", X_image, 10)
		
		with tf.name_scope('dropout_parameter'):
			keep_prob = tf.placeholder(tf.float32, name="keep_prob")
		
		tf.summary.scalar('dropout_keep_probability', keep_prob)
		
		for i in range(_NUM_GPUS):
			with tf.device('/gpu:{}'.format(i)):
				with tf.name_scope('%s_%d' % ('tower', i)) as scope:
					_x_image = X_image[i * _BATCH_SIZE: (i + 1) * _BATCH_SIZE]
					_y = Y[i * _BATCH_SIZE: (i + 1) * _BATCH_SIZE]
					
					logits = ResNet(_x_image, keep_prob, reuse_vars)
					
					with tf.name_scope('total'):
						y_pred_cls = tf.argmax(logits, axis=1, name="y_pred_cls")
						
						loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=_y),
						                      name="loss")
						if i == 0:
							correct_prediction = tf.equal(y_pred_cls, tf.argmax(_y, axis=1), name="correct_predictions")
							accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="accuracy")
					
					if i == 0:
						tf.summary.scalar("loss", loss)
						tf.summary.scalar("accuracy", accuracy)
					# tf.summary.scalar("correct_predictions", correct_prediction)
					
					with tf.name_scope('train'):
						optimizer = tf.train.AdamOptimizer(1e-3, beta1=0.9, beta2=0.999, epsilon=1e-08,
						                                   name="AdamOptimizer")
						grads = optimizer.compute_gradients(loss)
						tower_grads.append(grads)
					
					reuse_vars = True
		
		avg_grads = average_gradients(tower_grads)
		train_op = optimizer.apply_gradients(avg_grads)
	
	return X, Y, loss, train_op, correct_prediction, accuracy, y_pred_cls, keep_prob


def get_data_set(name="train"):
	x = None
	y = None
	
	maybe_download_and_extract()
	
	folder_name = "cifar_10"
	
	f = open(tmp_path + 'data_set/' + folder_name + '/batches.meta', 'rb')
	f.close()
	
	if name is "train":
		for i in range(5):
			f = open(tmp_path + 'data_set/' + folder_name + '/data_batch_' + str(i + 1), 'rb')
			datadict = pickle.load(f, encoding="latin1")
			f.close()
			
			_X = datadict["data"]
			_Y = datadict['labels']
			
			_X = np.array(_X, dtype=float) / 255.0
			_X = _X.reshape([-1, 3, 32, 32])
			_X = _X.transpose([0, 2, 3, 1])
			_X = _X.reshape(-1, 32 * 32 * 3)
			
			if x is None:
				x = _X
				y = _Y
			else:
				x = np.concatenate((x, _X), axis=0)
				y = np.concatenate((y, _Y), axis=0)
	
	elif name is "test":
		f = open(tmp_path + 'data_set/' + folder_name + '/test_batch', 'rb')
		datadict = pickle.load(f, encoding="latin1")
		f.close()
		
		x = datadict["data"]
		y = np.array(datadict['labels'])
		
		x = np.array(x, dtype=float) / 255.0
		x = x.reshape([-1, 3, 32, 32])
		x = x.transpose([0, 2, 3, 1])
		x = x.reshape(-1, 32 * 32 * 3)
	
	return x, dense_to_one_hot(y)


def dense_to_one_hot(labels_dense, num_classes=10):
	num_labels = labels_dense.shape[0]
	index_offset = np.arange(num_labels) * num_classes
	labels_one_hot = np.zeros((num_labels, num_classes))
	labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
	
	return labels_one_hot


def _print_download_progress(count, block_size, total_size):
	"""
	
	:param count:
	:param block_size:
	:param total_size:
	:return:
	"""
	pct_complete = float(count * block_size) / total_size
	msg = "\r- Download progress: {0:.1%}".format(pct_complete)
	sys.stdout.write(msg)
	sys.stdout.flush()


def maybe_download_and_extract():
	"""
	Download the dataset if needed.
	"""
	
	main_directory = tmp_path + "data_set/"
	cifar_10_directory = main_directory + "cifar_10/"
	if not os.path.exists(main_directory) or not os.path.exists(cifar_10_directory):
		os.makedirs(main_directory)
		
		url = "http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
		filename = url.split('/')[-1]
		file_path = os.path.join(main_directory, filename)
		zip_cifar_10 = file_path
		file_path, _ = urlretrieve(url=url, filename=file_path, reporthook=_print_download_progress)
		
		print()
		print("Download finished. Extracting files.")
		if file_path.endswith(".zip"):
			zipfile.ZipFile(file=file_path, mode="r").extractall(main_directory)
		elif file_path.endswith((".tar.gz", ".tgz")):
			tarfile.open(name=file_path, mode="r:gz").extractall(main_directory)
		print("Done.")
		
		os.rename(main_directory + "./cifar-10-batches-py", cifar_10_directory)
		os.remove(zip_cifar_10)


def train(epoch):
	"""
	Train the network.
	
	:param epoch: The current epoch
	:type epoch: int
	"""
	
	global tensorboard_train_counter
	total_batch = _BATCH_SIZE * _NUM_GPUS
	batch_count = int(math.ceil(len(train_x) / total_batch))
	for s in range(batch_count):
		batch_xs = train_x[s * total_batch: (s + 1) * total_batch]
		batch_ys = train_y[s * total_batch: (s + 1) * total_batch]
		
		start_time = time()
		summery, _, batch_loss, batch_acc = sess.run(
				[merged, optimizer, loss, accuracy],
				feed_dict={x: batch_xs, y: batch_ys, keep_prob: 0.5})
		duration = time() - start_time
		train_writer.add_summary(summery, global_step=tensorboard_train_counter)
		tensorboard_train_counter += 1
		
		if s % 10 == 0:
			percentage = int(round((s / batch_count) * 100))
			msg = "Epoch {}: step: {} , batch_acc = {} , batch loss = {}"
			print(msg.format(epoch, s, batch_acc, batch_loss))
	
	test_and_save(epoch)


def test_and_save(epoch):
	"""
	Test the current accuracy and update the global variable global_accuracy accordingly
	
	:param epoch: The current epoch
	:type epoch: int
	"""
	
	global global_accuracy, tensorboard_test_counter
	run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
	
	i = 0
	predicted_class = np.zeros(shape=len(test_x), dtype=np.int)
	while i < len(test_x):
		# run_metadata = tf.RunMetadata()
		j = min(i + _BATCH_SIZE, len(test_x))
		batch_xs = test_x[i:j, :]
		batch_ys = test_y[i:j, :]
		summary, predicted_class[i:j] = sess.run(
				[merged, y_pred_cls],
				feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1},
				options=run_options  # ,
				# run_metadata=run_metadata
		)
		# test_writer.add_run_metadata(run_metadata, "epoch{}:step{}".format(epoch, i))
		i = j
	
	correct = (np.argmax(test_y, axis=1) == predicted_class)
	acc = correct.mean() * 100
	correct_numbers = correct.sum()
	
	test_writer.add_summary(summary, global_step=tensorboard_test_counter)
	tensorboard_test_counter += 1
	mes = "\nEpoch {} - accuracy: {:.2f}% ({}/{})"
	print(mes.format((epoch + 1), acc, correct_numbers, len(test_x)))
	
	if global_accuracy != 0 and global_accuracy < acc:
		mes = "This epoch receive better accuracy: {:.2f} > {:.2f}. Saving session..."
		print(mes.format(acc, global_accuracy))
		global_accuracy = acc
		
		if not os.path.exists(save_folder):
			os.mkdir(save_folder)
		saver.save(sess, os.path.join(save_folder, save_file))
	
	elif global_accuracy == 0:
		global_accuracy = acc
	
	print("###########################################################################################################")


# GLOBALS
_NUM_CLASSES = 10

# PARAMS
_BATCH_SIZE = 128
_EPOCH = 5
_NUM_GPUS = 4

train_x, train_y = get_data_set("train")
test_x, test_y = get_data_set("test")
x, y, loss, optimizer, correct_prediction, accuracy, y_pred_cls, keep_prob = model()
global_accuracy = 0


merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter(tmp_path + 'tensorboard/hw2/train', sess.graph)
test_writer = tf.summary.FileWriter(tmp_path + 'tensorboard/hw2/test')

saver = tf.train.Saver()
save_path = 'saves/'
save_folder = os.path.join(save_path, pretty_time())
save_file = 'save.ckpt'
if not os.path.exists(save_path):
	os.mkdir(save_path)

tf.global_variables_initializer().run(session=sess)


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
	with open('total_parameters' + pretty_time() + '.txt', 'w') as f:
		f.write(str(total_parameters))




# input()


def main(args=None):
	global _EPOCH, _NUM_GPUS, save_folder
	# if tf.gfile.Exists(tmp_path + "tensorboard/hw2"):
	# 	tf.gfile.DeleteRecursively(tmp_path + "tensorboard/hw2")
	# tf.gfile.MakeDirs(tmp_path + "tensorbaord/hw2")
	
	# Setting the args
	if args is not None:
		if args.load:
			saves = glob.glob(save_path + "*")
			read_file = max(saves, key=os.path.getmtime, default=save_folder)
			if len(read_file) is 0:
				print('files to read not found. starting from the begining. ')
			else:
				print('restoring last check point')
				saver.restore(sess, os.path.join(read_file, 'save.ckpt'))
				
				save_folder = read_file
		
		_EPOCH = args.epochs
		_NUM_GPUS = args.gpus
	else:
		_EPOCH = 5
		_NUM_GPUS = 4
	
	get_total_parameters()
	
	start = time()
	for i in range(_EPOCH):
		print("\nEpoch: {0}/{1}\n".format((i + 1), _EPOCH))
		start_time = time()
		train(i)
		print('epoch %d took: %d time' % (i, time() - start_time))
	
	length = time() - start
	print("{0} Epoches took {1}sec. avg of {2}sec per epoch".format(_EPOCH, length, length / _EPOCH))


if __name__ == "__main__":
	main()

train_writer.close()
test_writer.close()
# sess.close()
