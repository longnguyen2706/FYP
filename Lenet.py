import numpy as np
import tensorflow as tf

def cnn_model_fn(features, labels, mode):

    #[batch_size, width, height, channel]
    #batch_size = -1 for dynamic batch size; channel =1 for BW image
    input_layer = tf.reshape(features, [-1,28,28, 1])

    #padding = same - the output tensor has same width and height with the input tensor
    #filter = 32: 32 filters to apply
    #output tensor size: [batch_size, 28,28, 32]
    conv1 = tf.layers.conv2d(
        inputs = input_layer,
        filters=32,
        kernel_size = [5,5],
        padding = "same",
        activation=tf.nn.relu
    )

    pool1 = tf.layers.max_pooling2d(inputs = conv1, pool_size =[2,2], stride =2 )

