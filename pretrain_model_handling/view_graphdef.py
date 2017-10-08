import tensorflow as tf
from tensorflow.python.platform import gfile

with tf.Session() as sess:
    model_filename ='/home/duclong002/pretrained_model/inception-resnet/inception-resnet_v2.pb'
    with gfile.FastGFile(model_filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        g_in = tf.import_graph_def(graph_def)
        for op in tf.get_default_graph().get_operations():
            print(str(op.name))
LOGDIR='/home/duclong002/pretrained_model/mobilenet/graph'
train_writer = tf.summary.FileWriter(LOGDIR)
train_writer.add_graph(sess.graph)