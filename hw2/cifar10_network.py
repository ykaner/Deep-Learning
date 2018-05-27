#! python3
from __future__ import absolute_import

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


def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)


def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)


def max_pool_2x2(x, name):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)


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
	
	conv1 = utils.tensorboard.conv2d_layer(x_image, [3, 3, 3, 32], layer_name="conv_1")
	
	conv1_1 = utils.tensorboard.conv2d_layer(conv1, [3, 3, 32, 64], layer_name="conv_1_1")
	
	pool = max_pool_2x2(conv1_1, name="pool")
	
	drop = tf.nn.dropout(pool, keep_prob, name="drop")
	
	conv2 = utils.tensorboard.conv2d_layer(drop, [3, 3, 64, 128], layer_name="conv_2")
	
	pool2 = max_pool_2x2(conv2, name="pool2")
	
	conv3 = utils.tensorboard.conv2d_layer(pool2, [2, 2, 128, 128], layer_name="conv_3")
	
	pool3 = max_pool_2x2(conv3, name="pool3")
	
	drop3 = tf.nn.dropout(pool3, keep_prob, name="drop3")
	
	flat = tf.reshape(drop3, [-1, 4 * 4 * 128], name="flat")
	
	with tf.variable_scope('fc_1'):
		fc = tf.nn.relu(tf.layers.dense(inputs=flat, units=1500, name="dense_layer"),
		                name="relu")  # , activation=tf.nn.relu)
		drop4 = tf.nn.dropout(fc, keep_prob, name="dropout")
	
	tf.summary.histogram("drop4", drop4)
	
	with tf.variable_scope('fc_2'):
		fc2 = tf.nn.relu(tf.layers.dense(inputs=drop4, units=1000, name="dense_layer"), name="fc")
		drop5 = tf.nn.dropout(fc2, keep_prob, name="dropout")
	
	tf.summary.histogram("drop5", drop5)
	
	with tf.name_scope('total'):
		softmax = tf.nn.softmax(tf.layers.dense(inputs=drop5, units=_NUM_CLASSES), name="softmax")
		y_pred_cls = tf.argmax(softmax, axis=1, name="y_pred_cls")
		
		loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=softmax, labels=y), name="loss")
		correct_prediction = tf.equal(y_pred_cls, tf.argmax(y, axis=1), name="correct_predictions")
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="accuracy")
	
	tf.summary.scalar("loss", loss)
	tf.summary.scalar("accuracy", accuracy)
	# tf.summary.scalar("correct_predictions", correct_prediction)
	
	with tf.name_scope('train'):
		optimizer = tf.train.AdamOptimizer(1e-4, beta1=0.9, beta2=0.999, epsilon=1e-08, name="AdamOptimizer").minimize(
			loss, name="train_step")
	
	return x, y, loss, optimizer, correct_prediction, accuracy, y_pred_cls, keep_prob


def get_data_set(name="train"):
	x = None
	y = None
	
	maybe_download_and_extract()
	
	folder_name = "cifar_10"
	
	f = open('c:/tmp/data_set/' + folder_name + '/batches.meta', 'rb')
	f.close()
	
	if name is "train":
		for i in range(5):
			f = open('c:/tmp/data_set/' + folder_name + '/data_batch_' + str(i + 1), 'rb')
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
		f = open('c:/tmp/data_set/' + folder_name + '/test_batch', 'rb')
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
	main_directory = "c:/tmp/data_set/"
	cifar_10_directory = main_directory + "cifar_10/"
	if not os.path.exists(main_directory):
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
	batch_size = int(math.ceil(len(train_x) / _BATCH_SIZE))
	for s in range(batch_size):
		batch_xs = train_x[s * _BATCH_SIZE: (s + 1) * _BATCH_SIZE]
		batch_ys = train_y[s * _BATCH_SIZE: (s + 1) * _BATCH_SIZE]
		
		start_time = time()
		summery, _, batch_loss, batch_acc = sess.run(
				[merged, optimizer, loss, accuracy],
				feed_dict={x: batch_xs, y: batch_ys, keep_prob2: 0.5})
		duration = time() - start_time
		train_writer.add_summary(summery, s)
		
		if s % 10 == 0:
			percentage = int(round((s / batch_size) * 100))
			msg = "step: {} , batch_acc = {} , batch loss = {}"
			print(msg.format(s, batch_acc, batch_loss))
	
	test_and_save(epoch)


def test_and_save(epoch):
	global global_accuracy
	run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
	run_metadata = tf.RunMetadata()
	
	i = 0
	predicted_class = np.zeros(shape=len(test_x), dtype=np.int)
	while i < len(test_x):
		j = min(i + _BATCH_SIZE, len(test_x))
		batch_xs = test_x[i:j, :]
		batch_ys = test_y[i:j, :]
		summary, predicted_class[i:j] = sess.run(
				[merged, y_pred_cls],
				feed_dict={x: batch_xs, y: batch_ys, keep_prob2: 1},
				options=run_options,
				run_metadata=run_metadata
		)
		i = j
	
	correct = (np.argmax(test_y, axis=1) == predicted_class)
	acc = correct.mean() * 100
	correct_numbers = correct.sum()
	
	test_writer.add_run_metadata(run_metadata, "epoch{}".format(i))
	test_writer.add_summary(summary, epoch)
	mes = "\nEpoch {} - accuracy: {:.2f}% ({}/{})"
	print(mes.format((epoch + 1), acc, correct_numbers, len(test_x)))
	
	if global_accuracy != 0 and global_accuracy < acc:
		mes = "This epoch receive better accuracy: {:.2f} > {:.2f}. Saving session..."
		print(mes.format(acc, global_accuracy))
		global_accuracy = acc
	
	elif global_accuracy == 0:
		global_accuracy = acc
	
	print("###########################################################################################################")


sess = tf.Session()

train_x, train_y = get_data_set("train")
test_x, test_y = get_data_set("test")
x, y, loss, optimizer, correct_prediction, accuracy, y_pred_cls, keep_prob2 = model()
global_accuracy = 0

# PARAMS
_BATCH_SIZE = 128
_EPOCH = 300

merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter('c:/tmp/hw2/train', sess.graph)
test_writer = tf.summary.FileWriter('c:/tmp/hw2/test')

tf.global_variables_initializer().run(session=sess)



total_parameters = 0
for variable in tf.trainable_variables():
	shape = variable.get_shape()
	variable_parameters = 1
	for dim in shape:
		variable_parameters *= dim.value
	total_parameters += variable_parameters
print(total_parameters)
input()


def main():
	if tf.gfile.Exists("c:/temp/hw2"):
		tf.gfile.DeleteRecursively("/temp/hw2")
	tf.gfile.MakeDirs("c:/temp/hw2")
	
	for i in range(_EPOCH):
		print("\nEpoch: {0}/{1}\n".format((i + 1), _EPOCH))
		train(i)


if __name__ == "__main__":
	main()

train_writer.close()
test_writer.close()
sess.close()
