import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist
import json
import codecs

# Load the fashion-mnist pre-shuffled train data and test data
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
print(x_train.shape,x_test.shape, y_train.shape, y_test.shape)

"""
print(x_train[0])
print(y_train[0])

plt.imshow(x_train[0])
plt.show()
"""
# Normalize data
x_train = x_train.astype('float32') / 255

# Plot images
# Convert data into 2d and get numpy arrays
temp_x_train = []
for i in range(x_train.shape[0]):
	temp_x_train.append(x_train[i].ravel())
x_train = np.array(temp_x_train)

D = 784
K = 10
step_size = 1e-0
reg = 1e-3
iteration = 10
num_samples = x_train.shape[0]
data_shape = (num_samples, D)

graph = tf.Graph()

with graph.as_default():
	x = tf.placeholder(tf.float32, data_shape, name='inputX')
	y_true = tf.placeholder(tf.int32, (num_samples, ))

	w = tf.Variable(tf.random_uniform([D, K], -1.0, 1.0))
	b = tf.Variable(tf.random_uniform([1, K], -1.0, 1.0))

	h = tf.add(tf.matmul(x, w), b)


	loss = tf.losses.softmax_cross_entropy(tf.one_hot(y_true, K), h)
	optimizer = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

	init = tf.global_variables_initializer()

with tf.Session(graph=graph) as sess:
	sess.run(init)
	feed_dict = {x:x_train, y_true:y_train}
	for i in range(iteration):
		opt, loss_val = sess.run([optimizer, loss], feed_dict= feed_dict)
		print(loss_val)

		summary_writer = tf.summary.FileWriter('./logs', graph=tf.get_default_graph())
	summary_writer.flush()