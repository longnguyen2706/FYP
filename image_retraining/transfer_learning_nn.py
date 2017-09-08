from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from datetime import datetime
import hashlib
import os.path
import random
import re
import sys
import tarfile

import numpy as np
from six.moves import urllib
import tensorflow as tf

from tensorflow.python.framework import graph_util
from tensorflow.python.framework import tensor_shape
from tensorflow.python.platform import gfile
from tensorflow.python.util import compat

FLAGS = None

def create_image_lists(image_dir, testing_percentage, validation_percentage):
    if not gfile.Exists(image_dir):
        tf.logging.error("image directory '" + image_dir + "' not found.")
        return None
    result = {}
    fileWalk = [x for x in gfile.Walk(image_dir)]
    sub_dirs = [x[0] for x in gfile.Walk(image_dir)]

    print(sub_dirs)


def main(_):
    create_image_lists(FLAGS.image_dir, FLAGS.testing_percentage, FLAGS.validation_percentage)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir',
                        type=str,
                        default='',
                        help='Path to folders of labeled images.'
    )
    parser.add_argument('--validation_percentage',
                        type=int,
                        default=10,
                        help='Percentage of images used for validation'
    )
    parser.add_argument('--testing_percentage',
                        type=int,
                        default=10,
                        help='Percentage of images used for testing'
    )


    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]]+unparsed)