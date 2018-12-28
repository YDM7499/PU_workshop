from tensorflow.python.framework.graph_util import convert_variables_to_constants
from tensorflow.python.saved_model import utils
from tensorflow.core.framework import graph_pb2
from tensorflow.core.protobuf import saver_pb2
from tensorflow.python.client import session
# from tensorflow.python.framework import graph_io
from tensorflow.python.framework import importer
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.framework import graph_util
from tensorflow.python.ops import math_ops
# from tensorflow.python.ops import variable
from tensorflow.python.platform import test
from tensorflow.python.training import saver as saver_lib
from tensorflow.contrib.session_bundle import exporter
from tensorflow.python.tools import freeze_graph
from tensorflow.contrib.tensorboard.plugins import projector

from keras.models import Sequential
from keras.layers import Dense, TimeDistributed, Input, Dropout
from keras.models import Model
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
from keras import backend as K
from keras.utils.np_utils import to_categorical
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist
import json
import codecs
import os


# Load the fashion-mnist pre-shuffled train data and test data
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
print(x_train.shape,x_test.shape, y_train.shape, y_test.shape)

# Normalize data
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
# Plot images
# Convert data into 2d and get numpy arrays
temp_x_train = []
for i in range(x_train.shape[0]):
	temp_x_train.append(x_train[i].ravel())
x_train = np.array(temp_x_train)

temp_x_test = []
for i in range(x_test.shape[0]):
	temp_x_test.append(x_test[i].ravel())
x_test = np.array(temp_x_test)

def tensorboard_service(model):	
	input_fld = './'

	output_node_names_of_input_network = "input_layer" #comma separated
	output_node_names_of_final_network = "hidden1" #comma separated
	output_graph_name = 'model.pb'

	output_fld = input_fld
	weight_file_path = 'final.model'

	num_output = len(output_node_names_of_input_network.split(','))
	pred_node_names = output_node_names_of_final_network.split(',')
	pred = [None]*num_output
	for i in range(num_output):
		pred[i] = tf.identity(model.output[i], name=pred_node_names[i])
		pred_ip	 = tf.identity(model.input, name='input')

	checkpoint_file = output_fld + "checkpoint"
	checkpoint_state_name = "checkpoint_state"
	input_graph_name = output_fld + 'blstm.pb'
	output_graph_path = os.path.join(output_fld, output_graph_name)
	keras_architecture = os.path.join(output_fld, 'keras_architecture_json')
	keras_weights = os.path.join(output_fld, 'keras_weights')

	sess = K.get_session()
	graph = tf.get_default_graph()
	input_graph_def = graph.as_graph_def()
	saver = tf.train.Saver()
	saver.save(sess, checkpoint_file, global_step=0, latest_filename='checkpoint_state')
	tf.train.write_graph(input_graph_def, output_fld, 'blstm.pb.ascii', as_text=True)
	tf.train.write_graph(input_graph_def, output_fld, 'blstm.pb', as_text=False)
	writer = tf.summary.FileWriter(output_fld, graph=graph)

	K._LEARNING_PHASE = tf.constant(0)

	input_saver_def_path = "" # deprecated
	input_binary = True

	restore_op_name = "save/restore_all"
	filename_tensor_name = "save/Const:0"

	clear_devices = False

	freeze_graph.freeze_graph(input_graph_name, input_saver_def_path,
							input_binary, checkpoint_file+'-0',
							output_node_names_of_final_network, restore_op_name,
							filename_tensor_name, output_graph_path,
							clear_devices, "", None)


#Define model
model = Sequential()
model.add(Dense(128, activation='relu', input_dim=784, name='input_layer'))
model.add(Dense(64, activation='relu', name='hidden1'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax', name='hidden2'))
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=20, batch_size=32, validation_data=(x_test,y_test))

#model.save('final.model')
#model.load_weights('final.model')
#tensorboard_service(model)
