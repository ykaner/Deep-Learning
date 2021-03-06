#! python3
from __future__ import absolute_import

import glob
import math
import os
import pickle
import sys
import tarfile
import urllib
from builtins import range
from datetime import datetime
from functools import reduce
from time import time
from urllib.request import urlretrieve

import numpy as np
import tensorflow as tf

from . import cifar10
from .. import utils
from hw2.cifar10_input import distorted_inputs

if os.name is "nt":
	tmp_path = "C:/tmp/"
else:
	tmp_path = "/tmp/"

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('train_dir', '/tmp/cifar10_train',
                       """Directory where to write event logs """
                       """and checkpoint.""")
tf.flags.DEFINE_integer('max_steps', 10000,
                        """Number of batches to run.""")
tf.flags.DEFINE_integer('num_gpus', 1,
                        """How many GPUs to use.""")
tf.flags.DEFINE_boolean('log_device_placement', False,
                        """Whether to log device placement.""")
tf.flags.DEFINE_integer("epochs", 50, """How many epochs to run.""")

tensorboard_train_counter = 0
tensorboard_test_counter = 0

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)


def pretty_time():
	return str(datetime.now().replace(microsecond=0)).replace(':', '-')


def max_pool_2x2(x, name):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)


def avg_pool_2x2(x, name):
	return tf.nn.avg_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)


#
# def tower_loss(scope, images, labels):
# 	"""
# 	Calculate the total loss on a single tower running the CIFAR model.
#
# 	:param scope: unique prefix string identifying the CIFAR tower, e.g. 'tower_0'
# 	:param images: Images. 4D tensor of shape [batch_size, height, width, 3].
# 	:param labels: Labels. 1D tensor of shape [batch_size].
# 	:returns: Tensor of shape [] containing the total loss for a batch of data
# 	"""
#
# 	# Build ResNet Graph.
# 	logits = cifar10.ResNet(images)
#
# 	# Build the portion of the Graph calculating the losses. Note that we will
# 	# assemble the total_loss using a custom function below.
# 	_ = cifar10.loss(logits, labels)
#
# 	# Assemble all of the losses for the current tower only.
# 	losses = tf.get_collection('losses', scope)
#
# 	# Calculate the total loss for the current tower.
# 	total_loss = tf.add_n(losses, name='total_loss')
#
# 	# Attach a scalar summary to all individual losses and the total loss; do the
# 	# same for the averaged version of the losses.
# 	for l in losses + [total_loss]:
# 		# Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
# 		# session. This helps the clarity of presentation on tensorboard.
# 		loss_name = re.sub('tower_[0-9]*/', '', l.op.name)
# 		tf.summary.scalar(loss_name, l)
#
# 	return total_loss


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


def ResNet(images, keep_prob, is_train=False, reuse=False):
	net_option = 'A'
	with tf.variable_scope('ResNet', reuse=reuse):
		conv0 = utils.tensorboard.conv2d_layer(images, [3, 3, 3, 16], layer_name="conv_0", batch_n=False, is_train=is_train)
		
		def conv1_act(out, name):
			return tf.nn.relu(out + conv0, name)
		
		conv1 = utils.tensorboard.conv2d_layer(conv0, [3, 3, 16, 16], layer_name="conv_1", batch_n=True,
		                                       is_train=is_train, act=tf.nn.relu)
		
		conv1_1 = utils.tensorboard.conv2d_layer(conv1, [3, 3, 16, 16], layer_name="conv_1_1", batch_n=True,
		                                         is_train=is_train, act=conv1_act)
		
		shortcut1_1 = conv1_1
		
		def conv1_2_act(out, name):
			return tf.nn.relu(out + shortcut1_1, name)
		
		conv1_2 = utils.tensorboard.conv2d_layer(conv1_1, [3, 3, 16, 16], layer_name="conv_1_2", batch_n=True,
		                                         is_train=is_train, act=tf.nn.relu)
		
		conv1_3 = utils.tensorboard.conv2d_layer(conv1_2, [3, 3, 16, 16], layer_name="conv_1_3", batch_n=True,
		                                         is_train=is_train, act=conv1_2_act)
				
		shortcut2 = utils.tensorboard.shortcut(conv1_3, [16, 32], layer_name='shortcut2', option=net_option)
		
		def conv2_act(out, name):
			return tf.nn.relu(out + shortcut2, name)
		
		conv2 = utils.tensorboard.conv2d_layer(conv1_3, [3, 3, 16, 32], layer_name="conv_2", strides=[1, 2, 2, 1],
		                                       batch_n=True, is_train=is_train,
		                                       act=tf.nn.relu)
		
		conv2_1 = utils.tensorboard.conv2d_layer(conv2, [3, 3, 32, 32], layer_name="conv_2_1", batch_n=True,
		                                         is_train=is_train, act=conv2_act)
		
		shortcut2_2 = conv2_1
		
		def conv2_2_act(out, name):
			return tf.nn.relu(out + shortcut2_2, name)
		
		conv2_2 = utils.tensorboard.conv2d_layer(conv2_1, [3, 3, 32, 32], layer_name="conv_2_2", batch_n=True,
		                                         is_train=is_train, act=tf.nn.relu)
		
		conv2_3 = utils.tensorboard.conv2d_layer(conv2_2, [3, 3, 32, 32], layer_name="conv_2_3", batch_n=True,
		                                         is_train=is_train, act=conv2_2_act)
		
		shortcut2_3 = conv2_3
		
		def conv2_3_act(out, name):
			return tf.nn.relu(out + shortcut2_3, name)
		
		conv2_4 = utils.tensorboard.conv2d_layer(conv2_3, [3, 3, 32, 32], layer_name="conv_2_4", batch_n=True,
		                                         is_train=is_train, act=tf.nn.relu)
		
		conv2_5 = utils.tensorboard.conv2d_layer(conv2_4, [3, 3, 32, 32], layer_name="conv_2_5", batch_n=True,
		                                         is_train=is_train, act=conv2_3_act)
		
		shortcut3 = utils.tensroboard.shortcut(conv2_5, [32, 64], layer_name='shortcut3', option=net_option)
		
		def conv3_act(out, name):
			return tf.nn.relu(out + shortcut3, name)
		
		conv3 = utils.tensorboard.conv2d_layer(conv2_5, [3, 3, 32, 64], layer_name="conv_3", strides=[1, 2, 2, 1],
		                                       batch_n=True, is_train=is_train,
		                                       act=tf.nn.relu)
		
		conv3_1 = utils.tensorboard.conv2d_layer(conv3, [3, 3, 64, 64], layer_name="conv_3_1", batch_n=True,
		                                         is_train=is_train, act=conv3_act)
		
		shortcut3_2 = conv3_1
		
		def conv3_2_act(out, name):
			return tf.nn.relu(out + shortcut3_2, name)
		
		conv3_2 = utils.tensorboard.conv2d_layer(conv3_1, [3, 3, 64, 64], layer_name="conv_3_2", batch_n=True,
		                                         is_train=is_train, act=tf.nn.relu)
		
		conv3_3 = utils.tensorboard.conv2d_layer(conv3_2, [3, 3, 64, 64], layer_name="conv_3_3", batch_n=True,
		                                         is_train=is_train, act=conv3_2_act)
		
		def conv3_3_act(out, name):
			return tf.nn.relu(out + conv3_3, name)
		
		conv3_4 = utils.tensorboard.conv2d_layer(conv3_3, [3, 3, 64, 64], layer_name="conv_3_4", batch_n=True,
		                                         is_train=is_train, act=tf.nn.relu)
		
		conv3_5 = utils.tensorboard.conv2d_layer(conv3_4, [3, 3, 64, 64], layer_name="conv_3_5", batch_n=True,
		                                         is_train=is_train, act=conv3_3_act)
		
		gap = tf.layers.average_pooling2d(conv3_5, [8, 8], [8, 8], padding='VALID', name='gap')
		
		flat = tf.reshape(gap, [-1, 64], name="flat")
		
		with tf.variable_scope('fc_1'):
			fc = tf.nn.relu(tf.layers.dense(inputs=flat, units=32, name="dense_layer"),
			                name="relu")  # , activation=tf.nn.relu)
			drop4 = tf.nn.dropout(fc, keep_prob, name="dropout")
		
		with tf.device('cpu:0'):
			tf.summary.histogram("drop4", drop4)
		
		logits = tf.nn.softmax(tf.layers.dense(inputs=drop4, units=_NUM_CLASSES), name="softmax")
	
	return logits


def model(inputs, labels):
	with tf.device('/cpu:0'):
		tower_grads = []
		reuse_vars = False
		
		predictions = []
		correct_predictions = []
		accuracies = []
		# with tf.name_scope('input'):
		# 	X = tf.placeholder(tf.float32, shape=[None, _IMAGE_SIZE * _IMAGE_SIZE * _IMAGE_CHANNELS], name='Input')
		# 	Y = tf.placeholder(tf.float32, shape=[None, _NUM_CLASSES], name='Output')
		
		# if X is None or Y is None:
		
		X, Y = inputs, labels
		
		# else:  # if given as a placeholder and need reshape
		with tf.name_scope('input_reshape'):
			X_image = tf.reshape(X, [-1, _IMAGE_SIZE, _IMAGE_SIZE, _IMAGE_CHANNELS], name='images')
		
		with tf.name_scope('dropout_parameter'):
			keep_prob = tf.placeholder(tf.float32, name="keep_prob")
		is_train = tf.placeholder(tf.bool, name='is_training')
		learning_rate = tf.placeholder(tf.float32, name='learning_rate')
		
		tf.summary.image("intput", X_image, 10)
		
		tf.summary.scalar('dropout_keep_probability', keep_prob)
		
		for i in range(_NUM_GPUS):
			with tf.device('/gpu:{}'.format(i)):
				with tf.name_scope('%s_%d' % ('tower', i)) as scope:
					_x_image = X_image[i * NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN: (i + 1) * NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN]
					_y = Y[i * NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN: (i + 1) * NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN]
					
					logits = ResNet(_x_image, keep_prob, is_train, reuse_vars)
					
					with tf.name_scope('total'):
						y_pred_cls = tf.argmax(logits, axis=1, name="y_pred_cls")
						
						predictions.append(y_pred_cls)
						loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=_y),
						                      name="loss")
						correct_prediction = tf.equal(y_pred_cls, tf.argmax(_y, axis=1), name="correct_predictions")
						correct_predictions.append(correct_prediction)
						accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="accuracy")
						accuracies.append(accuracy)
					
					with tf.device('/cpu:0'):
						tf.summary.scalar("loss", loss)
						tf.summary.scalar("accuracy", accuracy)
					# tf.summary.scalar("correct_predictions", correct_prediction)
					
					with tf.name_scope('train'):
						# optimizer = tf.train.AdamOptimizer(5e-4, beta1=0.9, beta2=0.999, epsilon=1e-08,
						#                                    name="AdamOptimizer")
						
						# optimizer = tf.train.AdadeltaOptimizer(5e-4)
						
						optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9,
						                                       use_nesterov=True, name='MomentumOptimizer')
						grads = optimizer.compute_gradients(loss)
						tower_grads.append(grads)
					
					reuse_vars = True
		
		avg_grads = average_gradients(tower_grads)
		
		correct_predictions = tf.concat(correct_predictions, axis=0)
		predictions = tf.concat(predictions, axis=0)
		accuracy = tf.reduce_mean(accuracies)
		# update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		# with tf.control_dependencies(update_ops):
		train_op = optimizer.apply_gradients(avg_grads)
	
	return X, Y, loss, train_op, correct_predictions, accuracy, predictions, avg_grads, keep_prob, is_train, learning_rate


def get_data_set(name="train", distortion=False):
	x = None
	y = None
	
	maybe_download_and_extract()
	
	folder_name = "cifar_10_python"
	
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
		
		if distortion:
			main_directory = tmp_path + "data_set/"
			cifar_10_distortion_directory = main_directory + "cifar_10_distortion/"
			cifar_10_distortion_file = cifar_10_distortion_directory + 'data_batch_'
			
			if not os.path.exists(cifar_10_distortion_directory) \
					or not os.path.exists(cifar_10_distortion_file + '0'):
				if not os.path.exists(cifar_10_distortion_directory):
					os.makedirs(cifar_10_distortion_directory)
				
				x = np.reshape(x, [x.shape[0], _IMAGE_SIZE, _IMAGE_SIZE, _IMAGE_CHANNELS])
				
				flip_left_right = [np.flip(image, 1) for image in x]
				
				def crop_image(img):
					pad_size = 5
					paddings = [[pad_size, pad_size], [pad_size, pad_size], [0, 0]]
					img = np.pad(img, paddings, 'reflect')
					startx = np.random.randint(pad_size * 2)
					starty = np.random.randint(pad_size * 2)
					return img[starty:starty + _IMAGE_SIZE, startx:startx + _IMAGE_SIZE]
				
				crop = [crop_image(image) for image in x]
				
				def per_image_standardization(img):
					img_len = reduce(int.__mul__, img.shape)
					mean = np.mean(img)
					stddev = np.sum([np.power(xi - mean, 2.0) for xi in np.nditer(img.T)]) / img_len
					adjusted_stddev = max(stddev, 1.0 / np.sqrt(img_len))
					
					return (img - mean) / adjusted_stddev
				
				image_standardization = [per_image_standardization(image) for image in x]
				
				x = np.concatenate((x, flip_left_right, crop, image_standardization), axis=0)
				y = np.concatenate([y] * 4, axis=0)
				
				x = np.reshape(x, [x.shape[0], reduce(int.__mul__, x.shape[1:])])
				
				idx = np.arange(len(x))
				np.random.shuffle(idx)
				x = x[idx]
				y = y[idx]
				
				data_len = len(x)
				n_chuncks = 4
				chunck_size = math.ceil(data_len / n_chuncks)
				for i in range(n_chuncks):
					with open(cifar_10_distortion_file + str(i), 'wb') as f:
						pickle.dump([x[chunck_size * i: chunck_size * (i + 1)],
						             y[chunck_size * i: chunck_size * (i + 1)]],
						            f)
				
				with open(cifar_10_distortion_directory + 'batches.meta', 'wb') as f:
					pickle.dump({'n_chuncks': n_chuncks}, f)
			
			else:
				
				with open(cifar_10_distortion_directory + 'batches.meta', 'rb') as f:
					n_chuncks = pickle.load(f)['n_chuncks']
				
				x = None
				y = None
				for i in range(n_chuncks):
					with open(cifar_10_distortion_file + str(i), 'rb') as f:
						px, py = pickle.load(f)
						x = np.concatenate([x, px]) if x is not None else px
						y = np.concatenate([y, py]) if y is not None else py
				
				x = np.array(x)
				y = np.array(y)
	
	
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


# def maybe_download_and_extract():
# 	"""
# 	Download the dataset if needed.
# 	"""
#
# 	main_directory = tmp_path + "data_set/"
# 	cifar_10_directory = main_directory + "cifar_10/"
# 	if not os.path.exists(main_directory) or not os.path.exists(cifar_10_directory):
# 		os.makedirs(main_directory)
#
# 		url = "http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
# 		filename = url.split('/')[-1]
# 		file_path = os.path.join(main_directory, filename)
# 		zip_cifar_10 = file_path
# 		file_path, _ = urlretrieve(url=url, filename=file_path, reporthook=_print_download_progress)
#
# 		print()
# 		print("Download finished. Extracting files.")
# 		if file_path.endswith(".zip"):
# 			zipfile.ZipFile(file=file_path, mode="r").extractall(main_directory)
# 		elif file_path.endswith((".tar.gz", ".tgz")):
# 			tarfile.open(name=file_path, mode="r:gz").extractall(main_directory)
# 		print("Done.")
#
# 		os.rename(main_directory + "./cifar-10-batches-py", cifar_10_directory)
# 		os.remove(zip_cifar_10)


def maybe_download_and_extract():
	"""
	If needed, downloads and extracts the tarball from Alex's website.
	"""
	
	global dataset_directory
	
	if not os.path.exists(dataset_directory):
		os.makedirs(dataset_directory)
	filename = DATA_URL.split('/')[-1]
	filepath = os.path.join(dataset_directory, filename)
	if not os.path.exists(filepath):
		def _progress(count, block_size, total_size):
			sys.stdout.write(
					'\r>> Downloading %s %.1f%%' % (filename, float(count * block_size) / float(total_size) * 100.0))
			sys.stdout.flush()
		
		filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
		print()
		
		statinfo = os.stat(filepath)
		
		print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
	
	extracted_dir_path = os.path.join(dataset_directory, 'cifar-10-batches-bin')
	if not os.path.exists(extracted_dir_path):
		tarfile.open(filepath, 'r:gz').extractall(dataset_directory)
	
	dataset_directory = extracted_dir_path


def train(epoch):
	"""
	Train the network.
	
	:param epoch: The current epoch
	:type epoch: int
	"""
	
	def lr_dict(ep):
		if ep < 80:
			return 0.1
		elif ep < 120:
			return 0.01
		else:
			return 0.001
	
	with tf.Graph().as_default(), tf.device('/cpu:0'):
		# Create a variable to count the number of train() calls. This equals the
		# number of batches processed * FLAGS.num_gpus.
		global_step = tf.get_variable(
				'global_step', [],
				initializer=tf.constant_initializer(0), trainable=False)
		
		global tensorboard_train_counter
		batch_count = int(math.ceil(_DATA_LEN / _TOTAL_BATCH))
		
		for s in range(batch_count):
			# batch_xs = train_x[s * _TOTAL_BATCH: (s + 1) * _TOTAL_BATCH]
			# batch_ys = train_y[s * _TOTAL_BATCH: (s + 1) * _TOTAL_BATCH]
			
			start_time = time()
			summery, _, batch_loss, batch_acc = sess.run(
					[merged, optimizer, loss, accuracy],
					feed_dict={keep_prob: 0.7, is_train: True, learning_rate: lr_dict(epoch)})
			duration = time() - start_time
			train_writer.add_summary(summery, global_step=tensorboard_train_counter)
			tensorboard_train_counter += 1
			
			if s % 20 == 10:
				percentage = int(round((s / batch_count) * 100))
				msg = "Epoch {epoch:03d}: step: {step:03d} , batch_acc = {acc:02.5f} , batch loss = {loss:02.5f}"
				print(msg.format(epoch=epoch, step=s, acc=batch_acc, loss=batch_loss))
		
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
		j = min(i + _TOTAL_BATCH, len(test_x))
		batch_xs = test_x[i:j, :]
		batch_ys = test_y[i:j, :]
		summary, predicted_class[i:j] = sess.run(
				[merged, y_pred_cls],
				feed_dict={keep_prob: 1, is_train: False},
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
	
	with open(file_name_date, 'a' if os.path.exists(file_name_date) else 'w') as f:
		f.writelines(mes.format((epoch + 1), acc, correct_numbers, len(test_x)))
	
	if global_accuracy != 0 and global_accuracy < acc:
		mes = "This epoch receive better accuracy: {:.2f} > {:.2f}. Saving session..."
		print(mes.format(acc, global_accuracy))
		global_accuracy = acc
		
		if not os.path.exists(save_folder):
			os.mkdir(save_folder)
		saver.save(sess, os.path.join(save_folder, save_file))
		with open(os.path.join(save_folder, 'hyper_params'), 'wb') as f:
			hyper_params = {
				'epoch': epoch
			}
			pickle.dump(hyper_params, f)
	
	elif global_accuracy == 0:
		global_accuracy = acc
	
	print("###########################################################################################################")


# GLOBALS
_NUM_CLASSES = 10

_IMAGE_SIZE = 32
_IMAGE_CHANNELS = 3

DATA_URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'
dataset_directory = tmp_path + "data_set/" + "cifar_10/"

# PARAMS
_BATCH_SIZE = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 512 + 128
_EPOCH = 5
_NUM_GPUS = 4
_TOTAL_BATCH = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN * _NUM_GPUS
_DATA_LEN = 50000

maybe_download_and_extract()
# train_x, train_y = get_data_set("train", distortion=False)
train_x, train_y = distorted_inputs(data_dir=dataset_directory, batch_size=_BATCH_SIZE)
train_x = tf.reshape(train_x, [train_x.shape[0], reduce((lambda a, b: a * b), train_x.shape[1:])])


test_x, test_y = get_data_set("test")
x, y, loss, optimizer, correct_prediction, accuracy, y_pred_cls, avg_grads, keep_prob, is_train, learning_rate = model()
global_accuracy = 0

merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter(tmp_path + 'tensorboard/hw2/train', sess.graph)
test_writer = tf.summary.FileWriter(tmp_path + 'tensorboard/hw2/test')

file_name_date = str(_EPOCH) + '_' + str(datetime.now().year) + str(datetime.now().month) + str(
	datetime.now().day) + str(datetime.now().hour) + '.txt'

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
	global _EPOCH, _NUM_GPUS, _TOTAL_BATCH, save_folder, file_name_date
	# if tf.gfile.Exists(tmp_path + "tensorboard/hw2"):
	# 	tf.gfile.DeleteRecursively(tmp_path + "tensorboard/hw2")
	# tf.gfile.MakeDirs(tmp_path + "tensorbaord/hw2")
	
	was_epochs = 0
	# Setting the args
	if args is not None:
		if args.load:
			saves = glob.glob(save_path + "*")
			read_file = max(saves, key=os.path.getmtime, default=save_folder)
			if len(read_file) is 0:
				print('files to read not found. starting from the begining. ')
			else:
				print('restoring last check point')
				
				with open(os.path.join(read_file, 'hyper_params'), 'rb') as f:
					hyper_params = pickle.load(f)
				was_epochs = hyper_params['epoch']
				
				saver.restore(sess, os.path.join(read_file, 'save.ckpt'))
				
				save_folder = read_file
		
		_EPOCH = args.epochs
		_NUM_GPUS = args.gpus
		
		file_name_date = str(_EPOCH) + '_' + str(datetime.now().year) + str(datetime.now().month) + str(
				datetime.now().day) + str(datetime.now().hour) + '.txt'
	else:
		_EPOCH = 5
		_NUM_GPUS = 4
	
	_TOTAL_BATCH = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN * _NUM_GPUS
	
	print('calculating parameters')
	get_total_parameters()
	
	start = time()
	for i in range(was_epochs, _EPOCH):
		print("\nEpoch: {0:03}/{1:03}\n".format((i + 1), _EPOCH))
		start_time = time()
		train(i)
		print('Epoch {epoch:04d} took: {sec:02.4f} seconds'.format(epoch=i, sec=(time() - start_time)))
	
	length = time() - start
	print("{0} Epochs took {1}sec. avg of {2}sec per epoch".format(_EPOCH, length, length / _EPOCH))


if __name__ == "__main__":
	main()

train_writer.close()
test_writer.close()
# sess.close()
