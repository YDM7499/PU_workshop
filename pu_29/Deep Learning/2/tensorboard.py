import keras
from keras import backend as K
import tensorflow as tf
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
import os


def tensorboard_service(model):
	output_node_names_of_input_network = "hidden1" #comma separated
	output_node_names_of_final_network = "hidden3" #comma separated
	output_graph_name = 'model.pb'

	output_fld = './tensorboard'
	weight_file_path = 'model.hdf5'

	
	num_output = len(output_node_names_of_input_network.split(','))
	pred_node_names = output_node_names_of_final_network.split(',')
	pred = [None]*num_output
	for i in range(num_output):
		pred[i] = tf.identity(model.output[i], name=pred_node_names[i])
		pred_ip	 = tf.identity(model.input, name='input')
	
	checkpoint_file = output_fld + "checkpoint"
	checkpoint_state_name = "checkpoint_state"
	input_graph_name = output_fld + 'model.pb'
	output_graph_path = os.path.join(output_fld, output_graph_name)
	keras_architecture = os.path.join(output_fld, 'keras_architecture_json')
	keras_weights = os.path.join(output_fld, 'keras_weights')

	sess = K.get_session()
	graph = tf.get_default_graph()
	input_graph_def = graph.as_graph_def()
	saver = tf.train.Saver()
	saver.save(sess, checkpoint_file, global_step=0, latest_filename='checkpoint_state')
	tf.train.write_graph(input_graph_def, output_fld, 'model.pb.ascii', as_text=True)
	tf.train.write_graph(input_graph_def, output_fld, 'model.pb', as_text=False)
	writer = tf.summary.FileWriter(output_fld, graph=graph)
	
	"""
	config = projector.ProjectorConfig()
	embedding_conf = config.embeddings.add()
	embedding_conf.tensor_name = "hidden1"
	embedding_conf.metadata_path = os.path.join(output_fld, "metadata.tsv")
	projector.visualize_embeddings(writer, config)
	"""
	
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


