import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim
from slim.nets.inception_resnet_v2 import inception_resnet_v2, inception_resnet_v2_arg_scope
import os

checkpoints_dir = '/home/long/pretrained_model/inception-resnet/'
summaries_dir = '/home/long/pretrained_model/summaries/inception-resnet/'
image_size = inception_resnet_v2.default_image_size

with tf.Graph().as_default():
    X = tf.placeholder(tf.float32, shape = [None, image_size, image_size, 3])
    image = np.ones((1, image_size, image_size, 3), dtype=np.float32)
    with slim.arg_scope(inception_resnet_v2_arg_scope()):
        logits, _ = inception_resnet_v2(image, 1001, is_training=False)

    init_fn = slim.assign_from_checkpoint_fn(
        os.path.join(checkpoints_dir,'inception_resnet_v2_2016_08_30.ckpt'),
        slim.get_model_variables('InceptionResnetV2')
    )

    probabilities = tf.nn.softmax(logits)
    with tf.Session() as sess:
        init_fn(sess)
        np_image, probabilities = sess.run([X, probabilities], feed_dict={X: image})

merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter(summaries_dir + '/train',
                                     sess.graph)