import tensorflow as tf
from tensorflow.contrib import slim
from slim.nets import inception_resnet_v2
import numpy as np

image_size = inception_resnet_v2.default_image_size
channels = 3

def get_init_fn():
    checkpoint_exclude_scopes =['InceptionResnetV2/AuxLogits', 'InceptionResnetV2/Logits']

    exlusions = [scope.strip() for scope in checkpoint_exclude_scopes]

    variables_to_restore =[]
    for var in slim.get_model_variables():
        exluded = False
        for exlusion in exlusions:
            if var.op.name.startswith(exlusion):
                exluded = True
                break
        if not exluded:
            variables_to_restore.append(var)

    return slim.assign_from_checkpoint_fn(
        '/home/long/pretrained_model/inception-resnet/inception_resnet_v2_2016_08_30.ckpt',
        variables_to_restore
    )

with tf.Graph().as_default():
    image = np.ones((1, image_size, image_size, channels), np.uint8)

    with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
        logit, _ = inception_resnet_v2.inception_resnet_v2(image, 1001, False)
        optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
        train_op = slim.learning.create_train_op(logit, optimizer)





    slim.learning.train(train_op,
                        init_fn=get_init_fn(),
                        number_of_steps=2)