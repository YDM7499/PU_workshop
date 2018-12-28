import tensorflow as tf
import numpy as  np

graph = tf.Graph()


#Define model
with graph.as_default():
	x = tf.placeholder(tf.float32, shape=(2,2))
	w = tf.Variable(tf.random_uniform([2,2], -1.0, 1.0))
	h = tf.matmul(x, w)
	init = tf.global_variables_initializer()

#Create session
with tf.Session(graph = graph) as sess:
	sess.run(init)
	data = np.array([[1.,2.], [3.,4.]])
	print('data x:', data)
	feed_dict = {x:data}
	res = sess.run(h, feed_dict=feed_dict)
	print('W', sess.run(w))
	print(res)

	summary_writer = tf.summary.FileWriter('./logs', graph=tf.get_default_graph())
	summary_writer.flush()
	#Optimization
