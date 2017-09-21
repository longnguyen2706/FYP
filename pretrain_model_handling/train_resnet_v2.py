import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim
from slim.nets.resnet_v2 import resnet_v2, resnet_v2_152,resnet_arg_scope
import os

checkpoints_dir = '/home/long/pretrained_model/resnet_v2/'
summaries_dir = '/home/long/pretrained_model/summaries/resnet_v2'
image_size = resnet_v2.default_image_size

with tf.Graph().as_default():
    X = tf.placeholder(tf.float32, shape=[None, image_size, image_size, 3])
    image = np.ones((1, image_size, image_size, 3), dtype=np.float32)
    with slim.arg_scope(resnet_arg_scope()):
        logits, _ = resnet_v2_152(image, 1001, is_training=False)

    init_fn = slim.assign_from_checkpoint_fn(
        os.path.join(checkpoints_dir, 'resnet_v2_152.ckpt'),
        slim.get_model_variables('resnet_v2'))

    probabilities = tf.nn.softmax(logits)
    with tf.Session() as sess:
        init_fn(sess)
        np_image, probabilities = sess.run([X, probabilities], feed_dict={X: image})
        # with tf.name_scope('summaries'):
        #     tf.summary.scalar(probabilities)
        # probabilities = probabilities[0, 0:]
        # sorted_inds = [i[0] for i in sorted(enumerate(-probabilities), key=lambda x: x[1])]

    # names = imagenet.create_readable_names_for_imagenet_labels()
    # for i in range(5):
    #     index = sorted_inds[i]
    #     # Shift the index of a class name by one.
    #     print('Probability %0.2f%% => [%s]' % (probabilities[index] * 100, names[index + 1]))
    #
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter(summaries_dir + '/train',
                                      sess.graph)