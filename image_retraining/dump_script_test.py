import argparse
import sys
import tensorflow as tf

FLAGS = None

def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.logging.info('var1 and var2: %d %d', FLAGS.var1, FLAGS.var2)

    # with tf.Session() as sess:
    #     init = tf.global_variables_initializer()
    #     sess.run(init)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--var1',
        type=int,
        default=0,
        help='Variable 1'
    )
    parser.add_argument(
        '--var2',
        type=int,
        default=0,
        help='Variable 2'
    )

    FLAGS, unparsed = parser.parse_known_args()

    tf.app.run(main= main, argv=[sys.argv[0]]+unparsed)