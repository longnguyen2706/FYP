import tensorflow as tf
import tensorflow.contrib.slim as slim

from slim.nets.vgg import vgg_19, vgg_arg_scope
from slim.nets.inception_resnet_v2 import inception_resnet_v2, inception_resnet_v2_arg_scope
from slim.nets.resnet_v2 import  resnet_v2_152, resnet_v2, resnet_arg_scope
import numpy as np

height = 224
width = 224
channels = 3




image_size = vgg_19.default_image_size
X = tf.placeholder(tf.float32, shape = [None, image_size, image_size, channels])
with slim.arg_scope(vgg_arg_scope()):
    logits, end_points = vgg_19(X, num_classes=1000, is_training=False)
    print(end_points)
    # print(end_points.shape, end_points.type)
predictions = end_points['vgg_19/fc8']


saver = tf.train.Saver()

X_test = np.ones((1,image_size,image_size,channels)) # a fake image

# Execute the graph
with tf.Session() as sess:
    saver.restore (sess, '/home/duclong002/pretrained_model/vgg19/vgg_19.ckpt' )
    predictions_val = predictions.eval(feed_dict={X: X_test})
    tf.train.write_graph(sess.graph_def, '/home/duclong002/pretrained_model/vgg19/', 'vgg19.pbtxt', True)


    #
    # X = tf.placeholder(tf.float32, shape=[None, image_size, image_size, channels])
    # with slim.arg_scope(inception_resnet_v2_arg_scope()):
    #     logits, end_points = inception_resnet_v2(X, num_classes=1001, is_training=False)
    #     print(end_points)


    # predictions = end_points['Predictions']
    # with tf.Session() as sess:
    #     # saver.restore (sess, '/home/long/pretrained_model/vgg19/vgg_19.ckpt' )
    #     saver.restore(sess, '/home/long/pretrained_model/inception-resnet/inception_resnet_v2_2016_08_30.ckpt')
    #     predictions_val = predictions.eval(feed_dict={X: X_test})
    #     tf.train.write_graph(sess.graph_def, './', 'inception-resnet_v2.pb', False)



# image_size = resnet_v2.default_image_size
# X = tf.placeholder(tf.float32, shape = [None, image_size, image_size, channels])
# with slim.arg_scope(resnet_arg_scope()):
#     logits, end_points = resnet_v2_152(X, num_classes=1001, is_training=False)
#     print(end_points)
#     for op in tf.get_default_graph().get_operations():
#         print(str(op.name))
#         # print(end_points.shape, end_points.type)
# #predictions = end_points['vgg_19/fc8']
# predictions = end_points['predictions']
#
# saver = tf.train.Saver()
#
# X_test = np.ones((1,image_size,image_size,channels)) # a fake image
#
# # Execute the graph
# with tf.Session() as sess:
#     #saver.restore (sess, '/home/long/pretrained_model/vgg19/vgg_19.ckpt' )
#     saver.restore(sess, '/home/duclong002/pretrained_model/resnet_v2/resnet_v2_152.ckpt')
#     predictions_val = predictions.eval(feed_dict={X: X_test})
#     # for op in sess.:
#     #     print(str(op.name), " : ",op.eval(), "\n")
#     tf.train.write_graph(sess.graph_def, '/home/duclong002/pretrained_model/resnet_v2/', 'resnet_v2_152.pbtxt', True)