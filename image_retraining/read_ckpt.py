import numpy as np
import tensorflow as tf
import os

from tensorflow.contrib import slim
from tensorflow.python.platform import gfile

def create_model_info(architecture):
    architecture = architecture.lower()

    if architecture == 'inception_v3':
        bottleneck_tensor_name = 'pool_3/_reshape:0'
        bottleneck_tensor_size = 2048
        input_weight = 299
        input_height = 299
        input_depth = 3
        resized_input_tensor_name = 'Mul:0'
        model_file_name = 'classify_image_graph_def.pb'
        input_mean = 128
        input_std = 128


def create_model_graph(model_info):
    with tf.Graph().as_default() as graph:
        model_path = os.path.join(FLAGS.model_dir, model_info['model_file_name'])
        with gfile.FastGfile(model_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            bottleneck_tensor, resized_input_tensor = (tf.import_graph_def(
                graph_def,
                name='',
                return_elements=[
                    model_info['bottleneck_tensor_name'],
                    model_info['resized_input_tensor_name']
                ]
            ))

    return graph, bottleneck_tensor, resized_input_tensor

def get_init_fn():

