from __future__ import print_function

import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# consts

in_shape = [28, 28, 1]
out_shape = 10

learning_rate = 0.001
epochs = 15
batch_size = 100

SAME = 'SAME'

num_gpus = 1
gpu_names = ['/device:GPU:' + str(i) for i in range(num_gpus)]

def main():
		
	x = tf.placeholder(tf.float32, [None] + in_shape)
	y = tf.placeholder(tf.float32, [None, out_shape])
	drop_out_p = tf.placeholder(tf.float32)
	for d in gpu_names:
		with tf.device(d):
			with tf.device('/cpu:0'):
				filters = tf.Variable(tf.random_normal([5, 5, 1, 32]))
				conv_w1 = tf.Variable(tf.random_normal([32]))
				filters2 = tf.Variable(tf.random_normal([5, 5, 32, 64]))
				conv_w2 = tf.Variable(tf.random_normal([64]))
			
				number_of_neurons = 1024
				weights = [tf.Variable(tf.random_normal([64 * 28 * 28, number_of_neurons])),
						   tf.Variable(tf.random_normal([number_of_neurons, out_shape]))]
				
				biases = [tf.Variable(tf.random_normal([number_of_neurons])),
						  tf.Variable(tf.random_normal([out_shape]))]
				
			conv_layer = tf.nn.relu(tf.nn.conv2d(x, filters, [1, 1, 1, 1], SAME) + conv_w1)
			pool_layer = tf.nn.max_pool(conv_layer, [1, 2, 2, 1], [1, 1, 1, 1], SAME)
			
			conv_layer2 = tf.nn.relu(tf.nn.conv2d(pool_layer, filters2, [1, 1, 1, 1], SAME) + conv_w2)
			pool_layer2 = tf.nn.max_pool(conv_layer2, [1, 2, 2, 1], [1] * 4, SAME)
			flaten = tf.contrib.layers.flatten(pool_layer2)
			
			fully_connected = tf.nn.relu(tf.matmul(flaten, weights[0]) + biases[0])
			drop_out = tf.nn.dropout(fully_connected, drop_out_p)
			out = tf.nn.relu(tf.matmul(drop_out, weights[1] + biases[1]))
		
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out, labels=y))
	
	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
	
	var_init = tf.global_variables_initializer()
		
	with tf.Session() as sess:
		sess.run(var_init)
		
		for epoch in range(epochs):
			total_cost = 0
			batch_num = int(mnist.train.num_examples / batch_size)
			for i in range(batch_num):
				batch_x, batch_y = mnist.train.next_batch(batch_size)
				batch_x = np.reshape(batch_x, [batch_size] + in_shape)
				feed_dict = {x: batch_x,
							 y: batch_y,
							 drop_out_p: 0.5}
				_, c = sess.run([optimizer, cost], feed_dict=feed_dict)
				total_cost += c
			
			total_cost /= batch_num
			print('Epoch: ' + str(epoch) + ', cost: ' + str(total_cost))
		
		# testing
		corrects = tf.equal(tf.argmax(out, 1), tf.argmax(y, 1))
		acc = tf.reduce_mean(tf.cast(corrects, tf.float32))
		print('acc: ' + str(acc.eval(
			{
				x: np.reshape(mnist.test.images, [mnist.test.image.shape[0]] + in_shape),
				y: mnist.test.labels,
				drop_out_p: 1
			})))


if __name__ == '__main__':
	main()
