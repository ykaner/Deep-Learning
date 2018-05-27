#! python3
from __future__ import absolute_import

import datetime
import math
import os
import pickle
import sys
import tarfile
import zipfile
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
	return str(datetime.datetime.now().replace(microsecond=0)).replace(':', '-')


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


def model():
	_IMAGE_SIZE = 32
	_IMAGE_CHANNELS = 3
	_NUM_CLASSES = 10
	
	with tf.name_scope('input'):
		x = tf.placeholder(tf.float32, shape=[None, _IMAGE_SIZE * _IMAGE_SIZE * _IMAGE_CHANNELS], name='Input')
		y = tf.placeholder(tf.float32, shape=[None, _NUM_CLASSES], name='Output')
	
	with tf.name_scope('input_reshape'):
		x_image = tf.reshape(x, [-1, _IMAGE_SIZE, _IMAGE_SIZE, _IMAGE_CHANNELS], name='images')
	
	tf.summary.image("intput", x_image, 10)
	
	with tf.name_scope('dropout'):
		keep_prob = tf.placeholder(tf.float32, name="keep_prob")
	
	tf.summary.scalar('dropout_keep_probability', keep_prob)
	
	conv0 = utils.tensorboard.conv2d_layer(x_image, [3, 3, 3, 16], layer_name="conv_0")
	
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
	
	with tf.name_scope('total'):
		softmax = tf.nn.softmax(tf.layers.dense(inputs=flat, units=_NUM_CLASSES), name="softmax")
		y_pred_cls = tf.argmax(softmax, axis=1, name="y_pred_cls")
		
		loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=softmax, labels=y), name="loss")
		correct_prediction = tf.equal(y_pred_cls, tf.argmax(y, axis=1), name="correct_predictions")
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="accuracy")
	
	tf.summary.scalar("loss", loss)
	tf.summary.scalar("accuracy", accuracy)
	# tf.summary.scalar("correct_predictions", correct_prediction)
	
	with tf.name_scope('train'):
		optimizer = tf.train.AdamOptimizer(1e-3, beta1=0.9, beta2=0.999, epsilon=1e-08, name="AdamOptimizer").minimize(
				loss, name="train_step")
	
	return x, y, loss, optimizer, correct_prediction, accuracy, y_pred_cls, keep_prob


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
	pct_complete = float(count * block_size) / total_size
	msg = "\r- Download progress: {0:.1%}".format(pct_complete)
	sys.stdout.write(msg)
	sys.stdout.flush()


def maybe_download_and_extract():
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
	global tensorboard_train_counter
	batch_count = int(math.ceil(len(train_x) / _BATCH_SIZE))
	for s in range(batch_count):
		batch_xs = train_x[s * _BATCH_SIZE: (s + 1) * _BATCH_SIZE]
		batch_ys = train_y[s * _BATCH_SIZE: (s + 1) * _BATCH_SIZE]
		
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
		
		saver.save(sess, os.path.join(save_path, save_file))
	
	elif global_accuracy == 0:
		global_accuracy = acc
	
	print("###########################################################################################################")


train_x, train_y = get_data_set("train")
test_x, test_y = get_data_set("test")
x, y, loss, optimizer, correct_prediction, accuracy, y_pred_cls, keep_prob = model()
global_accuracy = 0

# PARAMS
_BATCH_SIZE = 128
_EPOCH = 50

merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter(tmp_path + 'tensorboard/hw2/train', sess.graph)
test_writer = tf.summary.FileWriter(tmp_path + 'tensorboard/hw2/test')

saver = tf.train.Saver()
save_path = './saves/'
save_file = pretty_time() + '.ckpt'
if not os.path.exists(save_path):
	os.mkdir(save_path)

tf.global_variables_initializer().run(session=sess)

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


def main(args):
	global _EPOCH
	# if tf.gfile.Exists(tmp_path + "tensorboard/hw2"):
	# 	tf.gfile.DeleteRecursively(tmp_path + "tensorboard/hw2")
	# tf.gfile.MakeDirs(tmp_path + "tensorbaord/hw2")
	
	if args.load:
		read_file = os.listdir(save_path)
		if len(read_file) is 0:
			print('files to read not found. starting from the begining. ')
		else:
			print('restoring last check point')
			read_file = read_file[-1]
			saver.restore(sess, read_file)
	
	_EPOCH = args.epochs
	
	for i in range(_EPOCH):
		print("\nEpoch: {0}/{1}\n".format((i + 1), _EPOCH))
		start_time = time()
		train(i)
		print('epoch %d took: %d time' % (i, time() - start_time))


if __name__ == "__main__":
	main()

train_writer.close()
test_writer.close()
# sess.close()
