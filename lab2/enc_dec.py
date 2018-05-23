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




