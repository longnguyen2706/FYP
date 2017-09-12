from __future__ import absolute_import
import argparse
import sys
import tensorflow as tf
import csv
#from image_retraining import save_to_csv
FLAGS = None


def save_to_csv(filename, data_arr):
    f = open(filename, 'a')
    with f:
        writer =csv.writer(f)
        for row in data_arr:
            writer.writerow(row)

def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.logging.info('var1 and var2: %f %f', FLAGS.var1, FLAGS.var2)
    print(FLAGS)
    save_to_csv(FLAGS.csvlogfile, [[FLAGS]])
    # with tf.Session() as sess:
    #     init = tf.global_variables_initializer()
    #     sess.run(init)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--var1',
        type=float,
        default=0,
        help='Variable 1'
    )
    parser.add_argument(
        '--var2',
        type=float,
        default=0,
        help='Variable 2'
    )
    parser.add_argument(
        '--csvlogfile',
        type=str,
        default='',
        help='Link to logfile.csv'
    )

    FLAGS, unparsed = parser.parse_known_args()

    tf.app.run(main= main, argv=[sys.argv[0]]+unparsed)