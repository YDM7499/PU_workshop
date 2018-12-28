import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist

# Load the fashion-mnist pre-shuffled train data and test data
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
print("x_train shape:", x_train.shape, "y_train shape:", y_train.shape)
exit()

# Convert data into 2d and get numpy arrays
temp_x_train = []
for i in range(x_train.shape[0]):
	temp_x_train.append(x_train[i].ravel())
x_train = np.array(temp_x_train)

n_features = x_train.shape[1]
n_classes = 10
weights_shape = (n_features, n_classes)
bias_shape = (1, n_classes)


graph = tf.Graph()
with graph.as_default():
	X = tf.placeholder(dtype=tf.float32)
	Y_true = tf.placeholder(dtype=tf.int32)

	W = tf.Variable(dtype=tf.float32, initial_value=tf.random_normal(weights_shape))  # Weights of the model
	b = tf.Variable(dtype=tf.float32, initial_value=tf.random_normal(bias_shape))

	Y_pred = tf.matmul(X, W)  + b

	loss_function = tf.losses.softmax_cross_entropy(tf.one_hot(Y_true, n_classes), Y_pred)
	optimizer = tf.train.GradientDescentOptimizer(0.001).minimize(loss_function)

with tf.Session(graph = graph) as sess:
	sess.run(tf.global_variables_initializer())
	for i in range(10000):
		result, loss = sess.run([optimizer, loss_function], {X: x_train, Y_true: y_train})
		print("Iteration:", i, "loss:", loss)
		"""
		if i % 100 == 0:
			print("Iteration {}:\tLoss={:.6f}".format(i, sess.run(loss_function, {X: x_test, Y_true: y_test})))
		"""
	#y_pred = sess.run(Y_pred, {X: x_test})
	W_final, b_final = sess.run([W, b])

	summary_writer = tf.summary.FileWriter('./logs', graph=tf.get_default_graph())
	summary_writer.flush()