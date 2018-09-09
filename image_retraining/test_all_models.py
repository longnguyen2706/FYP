from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import csv
from datetime import datetime
import hashlib
import os.path
import random
import re
import sys
import tarfile
from operator import itemgetter
import numpy as np
from six.moves import urllib
import tensorflow as tf

from tensorflow.python.framework import graph_util
from tensorflow.python.framework import tensor_shape
from tensorflow.python.platform import gfile
from tensorflow.python.util import compat
from sklearn.metrics import confusion_matrix

################## General Setting #####################
# This section is to set the general setting for all models, i.e.: log file dir, summary dir, image dir ...
# Set where to write the csv file to. The file contains the printout of model performance every few thousands steps
csv_log_directory = '/home/duclong002/retrain_logs/logfile/' + "test_" + "ensemble_4_" + "Hep_" + str(datetime.now()).replace(" ", "-") + ".csv"
# Set where to write the summary file to. The file contains the model logs in graph form  which can be visualized by tensorboard
summaries_directory = '/home/duclong002/retrain_logs/ensemble/' + "test_" + "ensemble_4_" + "Hep_" + str(datetime.now()).replace(" ", "-") + ".csv"

GENERAL_SETTING = {
    'bottleneck_dir': '/tmp/bottleneck',
    'logits_dir': 'tmp/logits',
    'checkpoint_dir': '/home/duclong002/checkpoints', # The directory to write the temporary checkpoints to
    'early_stopping_n_steps': 5,
    'eval_step_interval': 100,
    'final_tensor_name': 'final_result',
    'flip_left_right': False,
    'model_dir': '/home/duclong002/pretrained_model/', #The directory of the downloaded pretrained nets (ex: resnet, inceptionv3 ...)
    'output_labels': '/tmp/output_labels.txt',
    'print_misclassified_test_images': True,
    'random_brightness': 0, # No data augmentation
    'random_crop': 0,
    'random_scale': 0,
    'test_batch_size': -1, # In every test step, all images in  the test set are used to test the model performance
    'testing_percentage': 20, # Take 20% of the whole dataset at the begining to be the test set
    'validation_batch_size': -1, # In every validation step, all images in the validation set are used to validate the model performance
    'csvlogfile': csv_log_directory,
    'how_many_training_steps': 10000,
    'image_dir': '/home/duclong002/Dataset/JPEG_data/Hep_JPEG/', # !Important: set the dataset directory
    #'image_dir': '/home/duclong002/data_conversion/JPEG_data/Hela_JPEG/',
    'summaries_dir': summaries_directory
}

###################### Model Setting #######################
# This section set the best hyper-parameter for each model based on the recorded performance during the grid test
# Model settings for Hep
MODEL_SETTINGS = [
    {
        'architecture': ['resnet_v2', 'inception_v3', 'inception_resnet_v2'], # feature concat model
        'dropout_keep_prob': 0.7,
        'hidden_layer1_size': 200,
        'learning_rate': 0.1,
        'learning_rate_decay': 0.66,
        'train_batch_size': 50,
        'test_accuracies': []
    },

    {
        'architecture': ['inception_v3'],
        'dropout_keep_prob': 0.7,
        'hidden_layer1_size': 150,
        'learning_rate': 0.1,
        'learning_rate_decay': 0.33,
        'train_batch_size': 30,
        'test_accuracies': []

    },

    {
        'architecture': ['resnet_v2'],
        'dropout_keep_prob': 0.7,
        'hidden_layer1_size': 150,
        'learning_rate': 0.05,
        'learning_rate_decay': 0.66,
        'train_batch_size': 30,
        'test_accuracies': []
    },

    {
        'architecture': ['inception_resnet_v2'],
        'dropout_keep_prob': 0.8,
        'hidden_layer1_size': 50,
        'learning_rate': 0.03,
        'learning_rate_decay': 0.33,
        'train_batch_size': 30,
        'test_accuracies': []
    }

]

# Model settings for PAP_smear dataset
# MODEL_SETTINGS = [
#     {
#         'architecture': ['resnet_v2', 'inception_v3', 'inception_resnet_v2'], # feature concat model
#         'dropout_keep_prob': 0.7,
#         'hidden_layer1_size': 200,
#         'learning_rate': 0.1,
#         'learning_rate_decay': 0.66,
#         'train_batch_size': 50,
#         'test_accuracies': []
#     },
#
#     {
#         'architecture': ['inception_v3'],
#         'dropout_keep_prob': 0.7,
#         'hidden_layer1_size': 150,
#         'learning_rate': 0.1,
#         'learning_rate_decay': 0.33,
#         'train_batch_size': 30,
#         'test_accuracies': []
#
#     }
#
#     {
#         'architecture': ['resnet_v2'],
#         'dropout_keep_prob': 0.7,
#         'hidden_layer1_size': 150,
#         'learning_rate': 0.05,
#         'learning_rate_decay': 0.66,
#         'train_batch_size': 30,
#         'test_accuracies': []
#     },
#
#     {
#         'architecture': ['inception_resnet_v2'],
#         'dropout_keep_prob': 0.8,
#         'hidden_layer1_size': 50,
#         'learning_rate': 0.03,
#         'learning_rate_decay': 0.33,
#         'train_batch_size': 30,
#         'test_accuracies': []
#     }
#
# ]


# Model settings for Hela dataset
# MODEL_SETTINGS = [
#     {
#         'architecture': ['resnet_v2', 'inception_v3', 'inception_resnet_v2'], # feature concat model
#         'dropout_keep_prob': 0.7,
#         'hidden_layer1_size': 50,
#         'learning_rate': 0.075,
#         'learning_rate_decay': 0.5,
#         'train_batch_size': 30,
#         'test_accuracies': []
#     },
#
#     {
#         'architecture': ['inception_v3'],
#         'dropout_keep_prob': 0.6,
#         'hidden_layer1_size': 50,
#         'learning_rate': 0.075,
#         'learning_rate_decay': 0.5,
#         'train_batch_size': 50,
#         'test_accuracies': []
#     },
#
#     {
#         'architecture': ['resnet_v2'],
#         'dropout_keep_prob': 0.7,
#         'hidden_layer1_size': 50,
#         'learning_rate': 0.1,
#         'learning_rate_decay': 0.5,
#         'train_batch_size': 50,
#         'test_accuracies': []
#     },
#
#     {
#         'architecture': ['inception_resnet_v2'],
#         'dropout_keep_prob': 0.6,
#         'hidden_layer1_size': 50,
#         'learning_rate': 0.1,
#         'learning_rate_decay': 0.33,
#         'train_batch_size': 100,
#         'test_accuracies': []
#     },
#
# ]

# These are all parameters that are tied to the particular model architecture
# we're using for Inception v3. These include things like tensor names and their
# sizes. If you want to adapt this script to work with another model, you will
# need to update these to reflect the values in the network you're using.
MAX_NUM_IMAGES_PER_CLASS = 2 ** 27 - 1  # ~134M


def save_to_csv(filename, data_arr):
    # Function to create a csv file that record the model performance.
    # If file does not exist, create new file. Else, append to the existing file
    if not os.path.exists(filename):
        f = open(filename, 'w')
    else:
        f = open(filename, 'a')

    with f:
        writer = csv.writer(f)
        for row in data_arr:
            writer.writerow(row)


def create_image_lists(image_dir):
    """Builds a list of training images from the file system.

    Analyzes the sub folders in the image directory, splits them into stable
    training, testing, and validation sets, and returns a data structure
    describing the lists of images for each label and their paths.

    Args:
      image_dir: String path to a folder containing subfolders of images.


    Returns:
      A dictionary containing an entry for each label subfolder, with images split
      into training, testing, and validation sets within each label.
    """
    if not gfile.Exists(image_dir):
        tf.logging.error("Image directory '" + image_dir + "' not found.")
        return None
    result = {}
    sub_dirs = [x[0] for x in gfile.Walk(image_dir)]
    # The root directory comes first, so skip it.
    is_root_dir = True
    for sub_dir in sub_dirs:
        if is_root_dir:
            is_root_dir = False
            continue
        extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
        file_list = []
        dir_name = os.path.basename(sub_dir)
        if dir_name == image_dir:
            continue
        tf.logging.info("Looking for images in '" + dir_name + "'")
        for extension in extensions:
            file_glob = os.path.join(image_dir, dir_name, '*.' + extension)
            file_list.extend(gfile.Glob(file_glob))
        if not file_list:
            tf.logging.warning('No files found')
            continue
        if len(file_list) < 20:
            tf.logging.warning(
                'WARNING: Folder has less than 20 images, which may cause issues.')
        elif len(file_list) > MAX_NUM_IMAGES_PER_CLASS:
            tf.logging.warning(
                'WARNING: Folder {} has more than {} images. Some images will '
                'never be selected.'.format(dir_name, MAX_NUM_IMAGES_PER_CLASS))
        label_name = re.sub(r'[^a-z0-9]+', ' ', dir_name.lower())

        # init the collection that holds 10 folds namely fold1, fold2 ... fold10
        fold_collection = {}
        for i in range(1, 11):
            fold_collection['fold%s' % i] = []

        for file_name in file_list:
            base_name = os.path.basename(file_name)
            # We want to ignore anything after '_nohash_' in the file name when
            # deciding which set to put an image in, the data set creator has a way of
            # grouping photos that are close variations of each other. For example
            # this is used in the plant disease data set to group multiple pictures of
            # the same leaf.
            hash_name = re.sub(r'_nohash_.*$', '', file_name)
            # This looks a bit magical, but we need to decide whether this file should
            # go into the training, testing, or validation sets, and we want to keep
            # existing files in the same set even if more files are subsequently
            # added.
            # To do that, we need a stable way of deciding based on just the file name
            # itself, so we do a hash of that and then use that to generate a
            # probability value that we use to assign it.
            hash_name_hashed = hashlib.sha1(compat.as_bytes(hash_name)).hexdigest()
            percentage_hash = ((int(hash_name_hashed, 16) %
                                (MAX_NUM_IMAGES_PER_CLASS + 1)) *
                               (100.0 / MAX_NUM_IMAGES_PER_CLASS))
            for i in range(1, 11):
                # 0 to <10: fold1 ... 90 to <100 fold10
                if (percentage_hash >= 10 * (i - 1) and percentage_hash < 10 * i):
                    fold_collection['fold%s' % i].append(base_name)
                # in case 100, add to fold10
                if (percentage_hash == 100):
                    fold_collection['fold%s' % 10].append(base_name)

        result[label_name] = {
            'dir': dir_name,
            'fold_collection': fold_collection
        }
    # print("cross validation folds and test images", result)
    return result


def get_image_path(image_lists, label_name, index, image_dir, category):
    """"Returns a path to an image for a label at the given index.

    Args:
      image_lists: Dictionary of training images for each label.
      label_name: Label string we want to get an image for.
      index: Int offset of the image we want. This will be moduloed by the
      available number of images for the label, so it can be arbitrarily large.
      image_dir: Root folder string of the subfolders containing the training
      images.
      category: Name string of set to pull images from - training, testing, or
      validation.

    Returns:
      File system path string to an image that meets the requested parameters.

    """
    if label_name not in image_lists:
        tf.logging.fatal('Label does not exist %s.', label_name)
    label_lists = image_lists[label_name]

    if category not in label_lists['fold_collection']:
        tf.logging.fatal('Category does not exist %s.', category)
    category_list = label_lists['fold_collection'][category]
    if not category_list:
        tf.logging.fatal('Label %s has no images in the category %s.',
                         label_name, category)
    mod_index = index % len(category_list)
    base_name = category_list[mod_index]
    sub_dir = label_lists['dir']
    full_path = os.path.join(image_dir, sub_dir, base_name)
    return full_path


def get_bottleneck_path(image_lists, label_name, index, bottleneck_dir,
                        category, architecture):
    """"Returns a path to a bottleneck file for a label at the given index.

    Args:
      image_lists: Dictionary of training images for each label.
      label_name: Label string we want to get an image for.
      index: Integer offset of the image we want. This will be moduloed by the
      available number of images for the label, so it can be arbitrarily large.
      bottleneck_dir: Folder string holding cached files of bottleneck values.
      category: Name string of set to pull images from - training, testing, or
      validation.
      architecture: The name of the model architecture.

    Returns:
      File system path string to an image that meets the requested parameters.
    """
    return get_image_path(image_lists, label_name, index, bottleneck_dir,
                          category) + '_' + architecture + '.txt'


def create_model_graph(model_info):
    """"Creates a graph from saved GraphDef file and returns a Graph object.

    Args:
      model_info: Dictionary containing information about the model architecture.

    Returns:
      Graph holding the trained Inception network, and various tensors we'll be
      manipulating.
    """
    with tf.Graph().as_default() as graph:
        model_path = os.path.join(GENERAL_SETTING['model_dir'], model_info['model_file_name'])
        # print("model_path", model_path)
        with gfile.FastGFile(model_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            bottleneck_tensor, resized_input_tensor = (tf.import_graph_def(
                graph_def,
                name='',
                return_elements=[
                    model_info['bottleneck_tensor_name'],
                    model_info['resized_input_tensor_name'],
                ]))
    return graph, bottleneck_tensor, resized_input_tensor


def ensure_dir_exists(dir_name):
    """Makes sure the folder exists on disk.

    Args:
      dir_name: Path string to the folder we want to create.
    """
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def create_bottleneck_file(bottleneck_path, image_lists, label_name, index,
                           image_dir, category, sess, jpeg_data_tensor,
                           decoded_image_tensor, resized_input_tensor,
                           bottleneck_tensor):
    """Create a single bottleneck file."""
    tf.logging.info('Creating bottleneck at ' + bottleneck_path)
    image_path = get_image_path(image_lists, label_name, index,
                                image_dir, category)
    if not gfile.Exists(image_path):
        tf.logging.fatal('File does not exist %s', image_path)
    image_data = gfile.FastGFile(image_path, 'rb').read()
    try:
        bottleneck_values = run_bottleneck_on_image(
            sess, image_data, jpeg_data_tensor, decoded_image_tensor,
            resized_input_tensor, bottleneck_tensor)
    except Exception as e:
        raise RuntimeError('Error during processing file %s (%s)' % (image_path,
                                                                     str(e)))
    bottleneck_string = ','.join(str(x) for x in bottleneck_values)
    with open(bottleneck_path, 'w') as bottleneck_file:
        bottleneck_file.write(bottleneck_string)


def get_or_create_bottleneck(sess, image_lists, label_name, index, image_dir,
                             category, bottleneck_dir, jpeg_data_tensor,
                             decoded_image_tensor, resized_input_tensor,
                             bottleneck_tensor, architecture):
    """Retrieves or calculates bottleneck values for an image.

    If a cached version of the bottleneck data exists on-disk, return that,
    otherwise calculate the data and save it to disk for future use.

    Args:
      sess: The current active TensorFlow Session.
      image_lists: Dictionary of training images for each label.
      label_name: Label string we want to get an image for.
      index: Integer offset of the image we want. This will be modulo-ed by the
      available number of images for the label, so it can be arbitrarily large.
      image_dir: Root folder string of the subfolders containing the training
      images.
      category: Name string of which set to pull images from - training, testing,
      or validation.
      bottleneck_dir: Folder string holding cached files of bottleneck values.
      jpeg_data_tensor: The tensor to feed loaded jpeg data into.
      decoded_image_tensor: The output of decoding and resizing the image.
      resized_input_tensor: The input node of the recognition graph.
      bottleneck_tensor: The output tensor for the bottleneck values.
      architecture: The name of the model architecture.

    Returns:
      Numpy array of values produced by the bottleneck layer for the image.
    """
    label_lists = image_lists[label_name]
    sub_dir = label_lists['dir']
    sub_dir_path = os.path.join(bottleneck_dir, sub_dir)
    ensure_dir_exists(sub_dir_path)
    bottleneck_path = get_bottleneck_path(image_lists, label_name, index,
                                          bottleneck_dir, category, architecture)
    # print("architecture", architecture, "bottleneck_path", bottleneck_path)
    if not os.path.exists(bottleneck_path):
        create_bottleneck_file(bottleneck_path, image_lists, label_name, index,
                               image_dir, category, sess, jpeg_data_tensor,
                               decoded_image_tensor, resized_input_tensor,
                               bottleneck_tensor)
    with open(bottleneck_path, 'r') as bottleneck_file:
        bottleneck_string = bottleneck_file.read()
    did_hit_error = False
    try:
        bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
    except ValueError:
        tf.logging.warning('Invalid float found, recreating bottleneck')
        did_hit_error = True
    if did_hit_error:
        create_bottleneck_file(bottleneck_path, image_lists, label_name, index,
                               image_dir, category, sess, jpeg_data_tensor,
                               decoded_image_tensor, resized_input_tensor,
                               bottleneck_tensor)
        with open(bottleneck_path, 'r') as bottleneck_file:
            bottleneck_string = bottleneck_file.read()
        # Allow exceptions to propagate here, since they shouldn't happen after a
        # fresh creation
        bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
    return bottleneck_values


def get_bottleneck(image_lists, label_name, index, image_dir,
                   category, bottleneck_dir, architecture):
    label_lists = image_lists[label_name]
    sub_dir = label_lists['dir']
    sub_dir_path = os.path.join(bottleneck_dir, sub_dir)
    ensure_dir_exists(sub_dir_path)
    bottleneck_path = get_bottleneck_path(image_lists, label_name, index,
                                          bottleneck_dir, category, architecture)
    # print("architecture", architecture, "bottleneck_path", bottleneck_path)

    with open(bottleneck_path, 'r') as bottleneck_file:
        bottleneck_string = bottleneck_file.read()
    did_hit_error = False
    try:
        bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
    except ValueError:
        tf.logging.warning('Invalid float found, recreating bottleneck')
        did_hit_error = True

        with open(bottleneck_path, 'r') as bottleneck_file:
            bottleneck_string = bottleneck_file.read()
        # Allow exceptions to propagate here, since they shouldn't happen after a
        # fresh creation
        bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
    return bottleneck_values


def cache_bottlenecks(sess, image_lists, image_dir, bottleneck_dir,
                      jpeg_data_tensor, decoded_image_tensor,
                      resized_input_tensor, bottleneck_tensor, architecture):
    """Ensures all the training, testing, and validation bottlenecks are cached.

    Because we're likely to read the same image multiple times (if there are no
    distortions applied during training) it can speed things up a lot if we
    calculate the bottleneck layer values once for each image during
    preprocessing, and then just read those cached values repeatedly during
    training. Here we go through all the images we've found, calculate those
    values, and save them off.

    Args:
      sess: The current active TensorFlow Session.
      image_lists: Dictionary of training images for each label.
      image_dir: Root folder string of the subfolders containing the training
      images.
      bottleneck_dir: Folder string holding cached files of bottleneck values.
      jpeg_data_tensor: Input tensor for jpeg data from file.
      decoded_image_tensor: The output of decoding and resizing the image.
      resized_input_tensor: The input node of the recognition graph.
      bottleneck_tensor: The penultimate output layer of the graph.
      architecture: The name of the model architecture.

    Returns:
      Nothing.
    """
    how_many_bottlenecks = 0
    ensure_dir_exists(bottleneck_dir)
    for label_name, label_lists in image_lists.items():
        for category in label_lists['fold_collection']:
            category_list = label_lists['fold_collection'][category]
            for index, unused_base_name in enumerate(category_list):
                get_or_create_bottleneck(
                    sess, image_lists, label_name, index, image_dir, category,
                    bottleneck_dir, jpeg_data_tensor, decoded_image_tensor,
                    resized_input_tensor, bottleneck_tensor, architecture)

                how_many_bottlenecks += 1
                if how_many_bottlenecks % 100 == 0:
                    tf.logging.info(
                        str(how_many_bottlenecks) + ' bottleneck files created.')


def get_random_cached_bottlenecks(image_lists, how_many, categories,
                                  image_dir, bottleneck_dir, architectures):
    """Retrieves bottleneck values for cached images.

    If no distortions are being applied, this function can retrieve the cached
    bottleneck values directly from disk for images. It picks a random set of
    images from the specified category.

    Args:
      sess: Current TensorFlow Session.
      image_lists: Dictionary of training images for each label.
      how_many: If positive, a random sample of this size will be chosen.
      If negative, all bottlenecks will be retrieved.
      category: Array of name string of which set to pull from - training, testing, or
      validation.
      bottleneck_dir: Folder string holding cached files of bottleneck values.
      image_dir: Root folder string of the subfolders containing the training
      images.
      jpeg_data_tensor: The layer to feed jpeg image data into.
      decoded_image_tensor: The output of decoding and resizing the image.
      resized_input_tensor: The input node of the recognition graph.
      bottleneck_tensor: The bottleneck output layer of the CNN graph.
      architecture: The name of the model architecture.

    Returns:
      List of bottleneck arrays, their corresponding ground truths, and the
      relevant filenames.
    """
    class_count = len(image_lists.keys())
    bottlenecks = []
    ground_truths = []
    filenames = []
    if how_many >= 0:
        # Retrieve a random sample of bottlenecks.
        for unused_i in range(how_many):
            label_index = random.randrange(class_count)
            label_name = list(image_lists.keys())[label_index]
            image_index = random.randrange(MAX_NUM_IMAGES_PER_CLASS + 1)
            # choose random fold from list of fold in training set
            category = random.choice(categories)
            image_name = get_image_path(image_lists, label_name, image_index,
                                        image_dir, category)
            merged_bottleneck = []
            merged_bottleneck_shape = 0
            for i in range(len(architectures)):

                architecture = architectures[i]
                bottleneck = get_bottleneck(
                    image_lists, label_name, image_index, image_dir, category,
                    bottleneck_dir, architecture)
                merged_bottleneck.append(bottleneck)
                merged_bottleneck_shape += len(bottleneck)

                ground_truth = np.zeros(class_count, dtype=np.float32)
                ground_truth[label_index] = 1.0
                if (i == len(architectures) - 1):
                    ground_truths.append(ground_truth)
                    filenames.append(image_name)

            # merged_bottleneck = np.reshape(merged_bottleneck, merged_bottleneck_shape)
            merged_bottleneck_reshape = []
            for sub_arr in merged_bottleneck:
                for item in sub_arr:
                    merged_bottleneck_reshape.append(item)
            bottlenecks.append(merged_bottleneck_reshape)


    else:
        # Retrieve all bottlenecks.
        for label_index, label_name in enumerate(image_lists.keys()):
            # choose random fold from list of fold in training set
            for category in categories:
                for image_index, image_name in enumerate(
                        image_lists[label_name]['fold_collection'][category]):
                    image_name = get_image_path(image_lists, label_name, image_index,
                                                image_dir, category)
                    merged_bottleneck = []
                    merged_bottleneck_shape = 0
                    for i in range(len(architectures)):

                        architecture = architectures[i]
                        bottleneck = get_bottleneck(
                            image_lists, label_name, image_index, image_dir, category,
                            bottleneck_dir, architecture)
                        merged_bottleneck.append(bottleneck)
                        merged_bottleneck_shape += len(bottleneck)

                        ground_truth = np.zeros(class_count, dtype=np.float32)
                        ground_truth[label_index] = 1.0
                        if (i == len(architectures) - 1):
                            ground_truths.append(ground_truth)
                            filenames.append(image_name)

                    # merged_bottleneck = np.reshape(merged_bottleneck, merged_bottleneck_shape)
                    merged_bottleneck_reshape = []
                    for sub_arr in merged_bottleneck:
                        for item in sub_arr:
                            merged_bottleneck_reshape.append(item)
                    bottlenecks.append(merged_bottleneck_reshape)
    return bottlenecks, ground_truths, filenames


def run_bottleneck_on_image(sess, image_data, image_data_tensor,
                            decoded_image_tensor, resized_input_tensor,
                            bottleneck_tensor):
    """Runs inference on an image to extract the 'bottleneck' summary layer.

    Args:
      sess: Current active TensorFlow Session.
      image_data: String of raw JPEG data.
      image_data_tensor: Input data layer in the graph.
      decoded_image_tensor: Output of initial image resizing and preprocessing.
      resized_input_tensor: The input node of the recognition graph.
      bottleneck_tensor: Layer before the final softmax.

    Returns:
      Numpy array of bottleneck values.
    """
    # First decode the JPEG image, resize it, and rescale the pixel values.
    resized_input_values = sess.run(decoded_image_tensor,
                                    {image_data_tensor: image_data})
    # Then run it through the recognition network.
    bottleneck_values = sess.run(bottleneck_tensor,
                                 {resized_input_tensor: resized_input_values})
    bottleneck_values = np.squeeze(bottleneck_values)
    return bottleneck_values


def save_graph_to_file(sess, graph, graph_file_name):
    output_graph_def = graph_util.convert_variables_to_constants(
        sess, graph.as_graph_def(), [GENERAL_SETTING['final_tensor_name']])
    with gfile.FastGFile(graph_file_name, 'wb') as f:
        f.write(output_graph_def.SerializeToString())
    return


def prepare_file_system():
    # Setup the directory we'll write summaries to for TensorBoard
    if tf.gfile.Exists(GENERAL_SETTING['summaries_dir']):
        tf.gfile.DeleteRecursively(GENERAL_SETTING['summaries_dir'])
    tf.gfile.MakeDirs(GENERAL_SETTING['summaries_dir'])
    return


def create_model_info(architecture):
    """Given the name of a model architecture, returns information about it.

    There are different base image recognition pretrained models that can be
    retrained using transfer learning, and this function translates from the name
    of a model to the attributes that are needed to download and train with it.

    Args:
      architecture: Name of a model architecture.

    Returns:
      Dictionary of information about the model, or None if the name isn't
      recognized

    Raises:
      ValueError: If architecture name is unknown.
    """
    architecture = architecture.lower()
    if architecture == 'inception_v3':
        # pylint: disable=line-too-long
        data_url = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
        # pylint: enable=line-too-long
        bottleneck_tensor_name = 'pool_3/_reshape:0'
        bottleneck_tensor_size = 2048
        input_width = 299
        input_height = 299
        input_depth = 3
        resized_input_tensor_name = 'Mul:0'
        model_file_name = 'classify_image_graph_def.pb'
        input_mean = 128
        input_std = 128

    elif architecture == 'resnet_v2':
        bottleneck_tensor_name = 'resnet_v2_152/pool5:0'
        bottleneck_tensor_size = 2048
        input_width = 224
        input_height = 224
        input_depth = 3
        resized_input_tensor_name = 'input:0'
        model_file_name = 'resnet_v2_152.pb'
        input_mean = 128
        input_std = 128

    elif architecture == 'resnet50':
        bottleneck_tensor_name = 'flatten_1/Reshape:0'
        bottleneck_tensor_size = 2048
        input_width = 224
        input_height = 224
        input_depth = 3
        resized_input_tensor_name = 'input_1:0'
        model_file_name = 'resnet50_keras.pb'
        input_mean = 128
        input_std = 128

    elif architecture == 'inception_resnet_v2':
        bottleneck_tensor_name = 'InceptionResnetV2/Logits/Flatten/Reshape:0'
        bottleneck_tensor_size = 1536
        input_width = 299
        input_height = 299
        input_depth = 3
        resized_input_tensor_name = 'InputImage:0'
        model_file_name = 'inception_resnet_v2_downloaded.pb'
        input_mean = 128
        input_std = 128

    elif architecture.startswith('mobilenet_'):
        parts = architecture.split('_')
        if len(parts) != 3 and len(parts) != 4:
            tf.logging.error("Couldn't understand architecture name '%s'",
                             architecture)
            return None
        version_string = parts[1]
        if (version_string != '1.0' and version_string != '0.75' and
                    version_string != '0.50' and version_string != '0.25'):
            tf.logging.error(
                """"The Mobilenet version should be '1.0', '0.75', '0.50', or '0.25',
        but found '%s' for architecture '%s'""",
                version_string, architecture)
            return None
        size_string = parts[2]
        if (size_string != '224' and size_string != '192' and
                    size_string != '160' and size_string != '128'):
            tf.logging.error(
                """The Mobilenet input size should be '224', '192', '160', or '128',
       but found '%s' for architecture '%s'""",
                size_string, architecture)
            return None
        if len(parts) == 3:
            is_quantized = False
        else:
            if parts[3] != 'quantized':
                tf.logging.error(
                    "Couldn't understand architecture suffix '%s' for '%s'", parts[3],
                    architecture)
                return None
            is_quantized = True
        data_url = 'http://download.tensorflow.org/models/mobilenet_v1_'
        data_url += version_string + '_' + size_string + '_frozen.tgz'
        bottleneck_tensor_name = 'MobilenetV1/Predictions/Reshape:0'
        bottleneck_tensor_size = 1001
        input_width = int(size_string)
        input_height = int(size_string)
        input_depth = 3
        resized_input_tensor_name = 'input:0'
        if is_quantized:
            model_base_name = 'quantized_graph.pb'
        else:
            model_base_name = 'frozen_graph.pb'
        model_dir_name = 'mobilenet_v1_' + version_string + '_' + size_string
        model_file_name = os.path.join(model_dir_name, model_base_name)
        input_mean = 127.5
        input_std = 127.5
    else:
        tf.logging.error("Couldn't understand architecture name '%s'", architecture)
        raise ValueError('Unknown architecture', architecture)

    return {
        'bottleneck_tensor_name': bottleneck_tensor_name,
        'bottleneck_tensor_size': bottleneck_tensor_size,
        'input_width': input_width,
        'input_height': input_height,
        'input_depth': input_depth,
        'resized_input_tensor_name': resized_input_tensor_name,
        'model_file_name': model_file_name,
        'input_mean': input_mean,
        'input_std': input_std,
        'architecture': architecture,
    }


def add_jpeg_decoding(input_width, input_height, input_depth, input_mean,
                      input_std):
    """Adds operations that perform JPEG decoding and resizing to the graph..

    Args:
      input_width: Desired width of the image fed into the recognizer graph.
      input_height: Desired width of the image fed into the recognizer graph.
      input_depth: Desired channels of the image fed into the recognizer graph.
      input_mean: Pixel value that should be zero in the image for the graph.
      input_std: How much to divide the pixel values by before recognition.

    Returns:
      Tensors for the node to feed JPEG data into, and the output of the
        preprocessing steps.
    """
    jpeg_data = tf.placeholder(tf.string, name='DecodeJPGInput')
    decoded_image = tf.image.decode_jpeg(jpeg_data, channels=input_depth)
    decoded_image_as_float = tf.cast(decoded_image, dtype=tf.float32)
    decoded_image_4d = tf.expand_dims(decoded_image_as_float, 0)
    resize_shape = tf.stack([input_height, input_width])
    resize_shape_as_int = tf.cast(resize_shape, dtype=tf.int32)
    resized_image = tf.image.resize_bilinear(decoded_image_4d,
                                             resize_shape_as_int)
    offset_image = tf.subtract(resized_image, input_mean)
    mul_image = tf.multiply(offset_image, 1.0 / input_std)
    return jpeg_data, mul_image


def add_final_training_ops(class_count, final_tensor_name, bottleneck_tensor_size, hidden_layer1_size, MODEL_SETTING):
    """Adds a new softmax and fully-connected layer for training.

    We need to retrain the top layer to identify our new classes, so this function
    adds the right operations to the graph, along with some variables to hold the
    weights, and then sets up all the gradients for the backward pass.

    The set up for the softmax and fully-connected layers is based on:
    https://www.tensorflow.org/versions/master/tutorials/mnist/beginners/index.html

    Args:
      class_count: Integer of how many categories of things we're trying to
      recognize.
      final_tensor_name: Name string for the new final node that produces results.
      bottleneck_tensor: The output of the main CNN graph.
      bottleneck_tensor_size: How many entries in the bottleneck vector.

    Returns:
      The tensors for the training and cross entropy results, and tensors for the
      bottleneck input and ground truth input.
    """
    with tf.name_scope('input'):
        bottleneck_input = tf.placeholder(
            tf.float32,
            shape=(None, bottleneck_tensor_size),
            name='BottleneckInputPlaceholder')

        ground_truth_input = tf.placeholder(tf.float32,
                                            [None, class_count],
                                            name='GroundTruthInput')
        keep_prob = tf.placeholder(tf.float32)
        global_step = tf.placeholder(tf.int32)
    tf.logging.info("Hidden layer 1 size = %d", hidden_layer1_size)
    # Organizing the following ops as `final_training_ops` so they're easier
    # to see in TensorBoard
    layer_name = 'final_training_ops'
    with tf.name_scope(layer_name):
        # FC hidden layer 1
        with tf.name_scope('hidden_weights_1'):
            initial_value = tf.truncated_normal(
                [bottleneck_tensor_size, hidden_layer1_size], stddev=0.01)
            hidden_weights_1 = tf.Variable(initial_value, name='hidden_weights_1')
            variable_summaries(hidden_weights_1)
        with tf.name_scope('hidden_biases_1'):
            hidden_biases_1 = tf.Variable(tf.zeros([hidden_layer1_size]), name='hidden_biases_1')
            variable_summaries(hidden_biases_1)
        with tf.name_scope('hidden_Wx_plus_b_1'):
            h_fc1 = tf.nn.relu(tf.matmul(bottleneck_input, hidden_weights_1) + hidden_biases_1)
            tf.summary.histogram('hidden layer 1', h_fc1)

        # Dropout
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        # Classifier layer
        with tf.name_scope('weights'):
            initial_value = tf.truncated_normal(
                [hidden_layer1_size, class_count], stddev=0.001)

            layer_weights = tf.Variable(initial_value, name='final_weights')

            variable_summaries(layer_weights)
        with tf.name_scope('biases'):
            layer_biases = tf.Variable(tf.zeros([class_count]), name='final_biases')
            variable_summaries(layer_biases)
        with tf.name_scope('Wx_plus_b'):
            logits = tf.matmul(h_fc1_drop, layer_weights) + layer_biases
            tf.summary.histogram('pre_activations', logits)

    final_tensor = tf.nn.softmax(logits, name=final_tensor_name)
    tf.summary.histogram('activations', final_tensor)

    with tf.name_scope('cross_entropy'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            labels=ground_truth_input, logits=logits)
        with tf.name_scope('total'):
            cross_entropy_mean = tf.reduce_mean(cross_entropy)
    tf.summary.scalar('cross_entropy', cross_entropy_mean)

    with tf.name_scope('train'):
        adaptive_learning_rate = tf.train.exponential_decay(MODEL_SETTING['learning_rate'], global_step, 1000,
                                                            MODEL_SETTING['learning_rate_decay'], staircase=True)

        # optimizer = tf.train.GradientDescentOptimizer(FLAGS.learning_rate)
        optimizer = tf.train.GradientDescentOptimizer(adaptive_learning_rate)
        train_step = optimizer.minimize(cross_entropy_mean)

    return (train_step, cross_entropy_mean, bottleneck_input, ground_truth_input,
            final_tensor, keep_prob, global_step, logits)


def add_evaluation_step(result_tensor, ground_truth_tensor):
    """Inserts the operations we need to evaluate the accuracy of our results.

    Args:
      result_tensor: The new final node that produces results.
      ground_truth_tensor: The node we feed ground truth data
      into.

    Returns:
      Tuple of (evaluation step, prediction).
    """
    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            prediction = tf.argmax(result_tensor, 1)
            correct_prediction = tf.equal(
                prediction, tf.argmax(ground_truth_tensor, 1))
        with tf.name_scope('accuracy'):
            evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', evaluation_step)
    return evaluation_step, prediction

def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def early_stopping_by_validation(train_validation_loss_results, early_stopping_n_steps):
    ''' Early stopping by validation performance
        An array of n most recent past validation performance (recorded every few hundred steps) will be passed to this function
        The function will then compare the current performance with the best known performance
        It returns False (stop training) if the current performance is the best one. Else return True (continue training)
    '''
    n_last_results = []
    if len(train_validation_loss_results) > early_stopping_n_steps:
        for i in range(len(train_validation_loss_results) - early_stopping_n_steps, len(train_validation_loss_results)):
            result = train_validation_loss_results[i]
            n_last_results.append(result)

        # print(n_last_results)

        descending_n_last_results = sorted(n_last_results, key=itemgetter('validation_accuracy'), reverse=True)
        if (n_last_results[0]['validation_accuracy'] == descending_n_last_results[0]['validation_accuracy']):
            return False, n_last_results[0]
        else:
            return True, n_last_results[0]
    else:
        return True, train_validation_loss_results[0]


def early_stopping_by_loss(train_cross_entropy_values, early_stopping_n_steps):
    '''
        Early stopping by loss
        An array of n most recent losses (recorded every few hundred steps) will be passed to this function
        The function will then compare the loss with the min loss in the known losses
        It returns False (stop training) if the current loss is the less than the min loss. Else return True (continue training)
    '''
    n_last_losses = []
    if len(train_cross_entropy_values) > early_stopping_n_steps:
        for i in range(len(train_cross_entropy_values) - early_stopping_n_steps, len(train_cross_entropy_values)):
            result = train_cross_entropy_values[i]
            n_last_losses.append(result)

        ascending_n_last_losses = sorted(n_last_losses, key=itemgetter('loss'), reverse=False)
        if (n_last_losses[0]['loss'] == ascending_n_last_losses[0]['loss']):
            return False, n_last_losses[0]
        else:
            return True, n_last_losses[0]
    else:
        return True, train_cross_entropy_values[0]


def get_checkpoint_path(step, training_fold_names, MODEL_SETTING):
    '''
    Get the checkpoint file that are generated during the training (.cpkt file that capture the network at a given step)
    '''
    fold_indexs = ""
    for training_fold_name in training_fold_names:
        fold_indexs = fold_indexs + training_fold_name[4:len(training_fold_name)]  # foldi-> i

    model = ''
    for architecture in MODEL_SETTING['architecture']:
        model = model + "_" + architecture

    checkpoint_name = "model_" + str(model) + "_step_" + str(step) + "_folds_" + str(fold_indexs) + "_lr_" + \
                      str(MODEL_SETTING['learning_rate']) + "_eval_inv_" + str(GENERAL_SETTING['eval_step_interval']) + \
                      "_train_b_" + str(MODEL_SETTING['train_batch_size']) + "_hidden1_" + str(
        MODEL_SETTING['hidden_layer1_size']) + \
                      "_dropout_" + str(MODEL_SETTING['dropout_keep_prob']) + "_early_s" + str(
        GENERAL_SETTING['early_stopping_n_steps'])
    checkpoint_path = os.path.join(GENERAL_SETTING['checkpoint_dir'], checkpoint_name)
    return checkpoint_path + '.cpkt'


def delete_and_update_checkpoint_arr(checkpoint_path_arr):
    '''
    Delete outdated checkpoints
    '''
    num_to_keep = GENERAL_SETTING['early_stopping_n_steps']
    if (len(checkpoint_path_arr) > num_to_keep):
        checkpoint_to_del = checkpoint_path_arr.pop(0)
        try:
            os.remove(checkpoint_to_del)
        except:
            tf.logging.info("Checkpoint dir to del does not exists")
    return checkpoint_path_arr


def delete_all_checkpoints(checkpoint_path_arr):
    for checkpoint_path in checkpoint_path_arr:
        checkpoint_path_metafile = checkpoint_path + ".meta"
        try:
            os.remove(checkpoint_path)
            os.remove(checkpoint_path_metafile)
        except:
            tf.logging.info("Checkpointdir to del does not exists")


def train_validation_two_decimal(train_accuracy, validation_accuracy):
    train_accuracy_two_decimal = int(train_accuracy * 10000) / 100
    validation_accuracy_two_decimal = int(validation_accuracy * 10000) / 100
    return train_accuracy_two_decimal, validation_accuracy_two_decimal


def get_last_layer_results(sess, image_lists, fold_names, architectures, bottleneck_input, ground_truth_input,
                           keep_prob, final_tensor, logits):
    last_layer_results = []

    # Save last layer result by feed-forward all images in the fold_names
    (all_bottlenecks, all_ground_truth, all_image_names) = get_random_cached_bottlenecks(
        image_lists, -1, fold_names,
        GENERAL_SETTING['image_dir'], GENERAL_SETTING['bottleneck_dir'], architectures)
    final_results, logits = sess.run(
        [final_tensor, logits],
        feed_dict={bottleneck_input: all_bottlenecks,
                   ground_truth_input: all_ground_truth,
                   keep_prob: 1.0,

                   })

    for i in range(len(all_image_names)):
        last_layer_result = {
            'image_name': all_image_names[i],
            'architectures': architectures,
            'ground_truth': all_ground_truth[i],
            'bottleneck_value': all_bottlenecks[i],
            'logit': logits[i],
            'final_result': final_results[i]
        }
        last_layer_results.append(last_layer_result)
    sorted_last_layer_results = sorted(last_layer_results, key=itemgetter('image_name'), reverse=False)
    return sorted_last_layer_results


def training_operation(image_lists, pretrained_model_infos, MODEL_SETTING, training_fold_names, testing_fold_names):
    '''
    This function is the most important function: do training, stop training by validation result.
    Recover the best checkpoint and doing testing.
    Record the result of training and testing to csv file
    '''
    train_cross_entropy_values = []

    checkpoint_path_arr = []

    architectures = []
    for i in range(len(pretrained_model_infos)):
        architecture = pretrained_model_infos[i]['architecture']
        architectures.append(architecture)

    tf.reset_default_graph()
    with tf.Session() as sess:
        bottleneck_tensor_size = 0
        for pretrained_model_info in pretrained_model_infos:
            bottleneck_tensor_size += pretrained_model_info['bottleneck_tensor_size']

        (train_step, cross_entropy, bottleneck_input, ground_truth_input,
         final_tensor, keep_prob, global_step, logits) = add_final_training_ops(
            len(image_lists.keys()), GENERAL_SETTING['final_tensor_name'], bottleneck_tensor_size,
            MODEL_SETTING['hidden_layer1_size'], MODEL_SETTING)

        # Create the operations we need to evaluate the accuracy of our new layer.
        evaluation_step, prediction = add_evaluation_step(
            final_tensor, ground_truth_input)

        # Merge all the summaries and write them out to the summaries_dir
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(GENERAL_SETTING['summaries_dir'] + '/train',
                                             sess.graph)

        validation_writer = tf.summary.FileWriter(
            GENERAL_SETTING['summaries_dir'] + '/validation')

        # Set up all our weights to their initial default values.
        init = tf.global_variables_initializer()
        sess.run(init)

        is_continue_training = True
        for i in range(GENERAL_SETTING['how_many_training_steps']):

            if (is_continue_training):
                (train_bottlenecks, train_ground_truth, _) = get_random_cached_bottlenecks(
                    image_lists, MODEL_SETTING['train_batch_size'], training_fold_names,
                    GENERAL_SETTING['image_dir'], GENERAL_SETTING['bottleneck_dir'], architectures)
                # print (i, "-", train_bottlenecks)

                train_summary, _ = sess.run(
                    [merged, train_step],
                    feed_dict={bottleneck_input: train_bottlenecks,
                               ground_truth_input: train_ground_truth,
                               keep_prob: MODEL_SETTING['dropout_keep_prob'],
                               global_step: i})

                train_writer.add_summary(train_summary, i)

                is_last_step = (i + 1 == GENERAL_SETTING['how_many_training_steps'])
                if (i % GENERAL_SETTING['eval_step_interval']) == 0 or is_last_step:
                    train_accuracy, cross_entropy_value = sess.run(
                        [evaluation_step, cross_entropy],
                        feed_dict={bottleneck_input: train_bottlenecks,
                                   ground_truth_input: train_ground_truth,
                                   keep_prob: 1.0,
                                   global_step: i
                                   })
                    tf.logging.info('%s: Step %d: Train accuracy = %.1f%%' %
                                    (datetime.now(), i, train_accuracy * 100))
                    tf.logging.info('%s: Step %d: Cross entropy = %f' %
                                    (datetime.now(), i, cross_entropy_value))

                    train_accuracy_two_decimal, validation_accuracy_two_decimal = train_validation_two_decimal(
                        train_accuracy, 0)
                    intermediate_result = ['', datetime.now(), i, train_accuracy * 100, cross_entropy_value, '',
                                           training_fold_names, testing_fold_names]
                    # Print the result to csvlogfile

                    save_to_csv(GENERAL_SETTING['csvlogfile'], [intermediate_result])

                    if i >= 0:
                        # Save a checkpoint
                        checkpoint_path = get_checkpoint_path(i, training_fold_names, MODEL_SETTING)
                        checkpoint_path_arr.append(checkpoint_path)

                        tf.train.Saver(write_version=tf.train.SaverDef.V1).save(sess, checkpoint_path)
                        checkpoint_path_arr = delete_and_update_checkpoint_arr(checkpoint_path_arr)
                        # print("checkpoint_path_arr", checkpoint_path_arr)


                        train_cross_entropy_value = {
                            'step': i,
                            'train_accuracy': train_accuracy_two_decimal,
                            'loss': cross_entropy_value,
                            'checkpoint_path': checkpoint_path
                        }
                        train_cross_entropy_values.append(train_cross_entropy_value)

                        # Early stopping condition check
                        is_continue_training, result = early_stopping_by_loss(train_cross_entropy_values,
                                                                              GENERAL_SETTING['early_stopping_n_steps'])
                        if not is_continue_training:
                            tf.logging.info("Early stopping. The best result is at %d steps: Train accuracy %.1f%%,"
                                            "Loss: %.f", result['step'],
                                            result['train_accuracy'],
                                            result['loss'])

                            early_stopping_logging_info = "The best final result is at " + str(
                                result['step']) + " steps:" + \
                                                          "Train accuracy: " + str(
                                result['train_accuracy']) + ", Loss: " + str(result['loss'])

                            early_stopping_result = ['', '', '', '', '', '', '', '', '', early_stopping_logging_info]
                            save_to_csv(GENERAL_SETTING['csvlogfile'], [early_stopping_result])

            else:
                break

        # We've completed all our training, so run a final test evaluation on
        # some new images we haven't used before.
        test_bottlenecks, test_ground_truth, test_filenames = (
            get_random_cached_bottlenecks(
                image_lists, GENERAL_SETTING['test_batch_size'], testing_fold_names,
                GENERAL_SETTING['image_dir'], GENERAL_SETTING['bottleneck_dir'], architectures))

        # restore the best checkpoint
        # ckpt = tf.train.get_checkpoint_state(result['checkpoint_path'])
        tf.train.Saver().restore(sess, result['checkpoint_path'])

        test_accuracy, predictions = sess.run(
            [evaluation_step, prediction],
            feed_dict={bottleneck_input: test_bottlenecks,
                       ground_truth_input: test_ground_truth, keep_prob: 1.0, global_step: i})
        tf.logging.info('Final test accuracy = %.1f%% (N=%d)' %
                        (test_accuracy * 100, len(test_bottlenecks)))

        misclassified_image_arr = []

        if GENERAL_SETTING['print_misclassified_test_images']:

            tf.logging.info('=== MISCLASSIFIED TEST IMAGES ===')
            for i, test_filename in enumerate(test_filenames):
                if predictions[i] != test_ground_truth[i].argmax():
                    tf.logging.info('%s  %s' %
                                    (test_filename,
                                     list(image_lists.keys())[predictions[i]]))
                    misclassified_image_arr.append((test_filename,
                                                    list(image_lists.keys())[predictions[i]]))

        # Print the result to csvlogfile
        final_result = ['', datetime.now(), '', '', '', '',
                        '', test_accuracy * 100, GENERAL_SETTING['summaries_dir'], misclassified_image_arr]
        save_to_csv(GENERAL_SETTING['csvlogfile'], [final_result])

        # Find all last layer results
        last_layer_train_results = get_last_layer_results(sess, image_lists, training_fold_names, architectures,
                                                          bottleneck_input, ground_truth_input, keep_prob, final_tensor,
                                                          logits)
        last_layer_test_results = get_last_layer_results(sess, image_lists, testing_fold_names, architectures,
                                                         bottleneck_input, ground_truth_input, keep_prob, final_tensor,
                                                         logits)
        delete_all_checkpoints(checkpoint_path_arr)

    sess.close()
    return (test_accuracy * 100), last_layer_train_results, last_layer_test_results


def add_naive_averaging_evaluation(class_count, is_logit):
    with tf.name_scope('naive_averaging_accuracy'):
        result_matrix = tf.placeholder(tf.float32,
                                       [None, class_count],
                                       name='result_matrix')
        ground_truth_matrix = tf.placeholder(tf.float32,
                                             [None, class_count],
                                             name='grouth_truth_matrix')
        with tf.name_scope('correct_prediction'):
            if not is_logit:
                naive_averaging_prediction = tf.argmax(result_matrix, 1)

            else:  # use logit result. Need to do softmax first
                after_softmax_result = tf.nn.softmax(result_matrix, name='after_softmax_result')
                naive_averaging_prediction = tf.argmax(after_softmax_result, 1)
            correct_prediction = tf.equal(
                naive_averaging_prediction, tf.argmax(ground_truth_matrix, 1)
            )
        with tf.name_scope('accuracy'):
            naive_averaging_eval_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return result_matrix, ground_truth_matrix, naive_averaging_eval_step, naive_averaging_prediction

def cal_conf_matrix(image_lists, prediction, ground_truth):

    labels = np.argmax(np.asarray(ground_truth), 1)
    print('labels: ', labels)
    print(image_lists.keys())
    conf = confusion_matrix(labels, prediction, list(set(labels)))

    conf_norm = []
    for row in (conf):
        if np.sum(row) == 0:
            conf_norm.append(row)
        else:
            conf_norm.append(row / np.sum(row))
    conf_norm = np.asarray(conf_norm)
    print('conf matrix', conf)
    print('conf norm', conf_norm)
    return conf_norm

def naive_averaging_result(image_lists, class_count, all_last_layer_test_results, is_logit):
    average_final_results = []
    ground_truths = []
    # print(all_last_layer_test_results)

    # base_learner_result_length = len(all_last_layer_test_results[0])
    # for i in range (0, base_learner_result_length):
    #     result = 0
    #     for base_learner_result in all_last_layer_test_results:
    #         result = result+ base_learner_result[i]['final_result']
    #     result = result/len(all_last_layer_test_results)
    #     average_result_matrix.append(result)
    for base_last_layer_test_result in all_last_layer_test_results:
        base_final_result = []
        for item in base_last_layer_test_result:
            # use the final result after softmax
            if not is_logit:
                base_final_result.append(item['final_result'])
            # use the result before softmax (logit)
            else:
                logit = item['logit']
                # logit_mean = sum(logit) / float(len(logit))
                # logit_stdev = np.std(logit, dtype=np.float32, ddof=1)
                # normalized_logit = [(float(i)-logit_mean)/logit_stdev for i in logit]
                # normalized_logit = [(float(i)-min(logit))/(max(logit)-min(logit)) for i in logit]
                normalized_logit = logit
                # print('logit', logit)
                # print ('normalized logit', normalized_logit)
                base_final_result.append(normalized_logit)

        average_final_results.append(base_final_result)
    # print('average final result shape', average_final_results, np.shape(average_final_results))
    average_final_results = np.mean(average_final_results, 0)
    print(average_final_results)
    # print('average final result shape after mean', average_final_results, np.shape(average_final_results))

    for item in all_last_layer_test_results[0]:
        ground_truths.append(item['ground_truth'])

    print("Length of the average final result and groundtruth: ",
          len(average_final_results), len(ground_truths))

    with tf.Session() as sess:
        result_matrix, ground_truth_matrix, naive_averaging_eval_step, naive_averaging_prediction = add_naive_averaging_evaluation(
            class_count, is_logit)
        test_accuracy, predictions = sess.run(
            [naive_averaging_eval_step, naive_averaging_prediction],
            feed_dict={result_matrix: average_final_results,
                       ground_truth_matrix: ground_truths})

        tf.logging.info('Final test accuracy = %.1f%% (N=%d)' %
                        (test_accuracy * 100, len(average_final_results)))


    print('prediction', predictions)
    print('ground truth', ground_truths)

    conf_norm = cal_conf_matrix(image_lists, predictions, ground_truths)
    return test_accuracy * 100, conf_norm
    sess.close()


def save_mean_and_std_result(result_arr, name):
    mean = sum(result_arr) / float(len(result_arr))
    std = np.std(result_arr, dtype=np.float32, ddof=1)
    print("average %s result" % str(name), mean)
    print("standard dev of %s" % str(name), std)
    result_to_save = ['', '', '', '', '', str(name), mean, std]
    save_to_csv(GENERAL_SETTING['csvlogfile'], [result_to_save])

def save_avg_conf(conf_arr, name, image_lists):
    conf_arr = np.asarray(conf_arr)
    print('conf arr shape: ', conf_arr.shape)
    avg_conf = np.mean(conf_arr, axis=0)
    print('avg conf: ', avg_conf)
    result_to_save = ['', '', '', '', '', str(name), conf_arr, avg_conf, image_lists.keys()]
    save_to_csv(GENERAL_SETTING['csvlogfile'], [result_to_save])

def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    prepare_file_system()

    # Print the FLAGS setting to logfile.csv
    save_to_csv(GENERAL_SETTING['csvlogfile'],
                [['Info', 'Step', 'Time', 'Train accuracy', 'Loss', '', "Train folds", "Test folds", "Test accuracy"]])
    SETTINGS = [GENERAL_SETTING, MODEL_SETTINGS]
    save_to_csv(GENERAL_SETTING['csvlogfile'], [SETTINGS])

    # Look at the folder structure, and create lists of all the images.
    image_lists = create_image_lists(GENERAL_SETTING['image_dir'])
    class_count = len(image_lists.keys())
    if class_count == 0:
        tf.logging.error('No valid folders of images found at ' + GENERAL_SETTING['image_dir'])
        return -1
    if class_count == 1:
        tf.logging.error('Only one valid folder of images found at ' +
                         GENERAL_SETTING['image_dir'] +
                         ' - multiple classes are needed for classification.')
        return -1

    # Train each base learner, get the test accuracy and last layer train and test result
    all_naive_ensemble_logits_test_accuracies = []
    all_naive_ensemble_softmax_test_accuracies = []
    three_bases_naive_ensemble_logits_test_accuracies = []
    three_bases_naive_ensemble_softmax_test_accuracies = []

    all_naive_averaging_with_logit_conf_arr =[]
    all_naive_averaging_with_softmax_conf_arr = []

    # In this section, we will perform the train and test for every model: 3 base models, feature concat model,
    # naive averaging models for 3 base models, naive averaging for all (3 base models + feature concat)
    # Initialize the net randomly, doing the training, early stopping based on cross-validation performance and then perform the test.
    # Repeat the process for 30 times, calculate the average result and the variance
    for index in range(0, 1):
        # Choosing 2 random folds inside 10 folds and create testing and training folds
        testing_fold_names = []
        [testing_fold_index_1, testing_fold_index_2] = random.sample(range(1, 11), 2)
        testing_fold_name_1 = 'fold' + str(testing_fold_index_1)
        testing_fold_name_2 = 'fold' + str(testing_fold_index_2)
        testing_fold_names.append(testing_fold_name_1)
        testing_fold_names.append(testing_fold_name_2)

        training_fold_names = []
        for training_fold_index in [v for v in range(1, 11) if
                                    (v != testing_fold_index_1 and v != testing_fold_index_2)]:
            training_fold_name = "fold" + str(training_fold_index)
            training_fold_names.append(training_fold_name)

        # For debug
        print('testing fold names: ', testing_fold_names, 'traning fold names: ', training_fold_names)

        all_last_layer_train_results = []
        all_last_layer_test_results = []
        three_bases_last_layer_train_results = []
        three_bases_last_layer_test_results = []
        testing_accuracy = []

        # looping through 3 base models and feature concat model
        for model_index in range(len(MODEL_SETTINGS)):
            MODEL_SETTING = MODEL_SETTINGS[model_index]
            save_to_csv(GENERAL_SETTING['csvlogfile'], [[MODEL_SETTING['architecture']]])

            pretrained_model_infos = []
            for i in range(len(MODEL_SETTING['architecture'])):
                architecture = MODEL_SETTING['architecture'][i]
                pretrained_model_info = create_model_info(architecture)
                pretrained_model_infos.append(pretrained_model_info)

                # cached bottleneck
                graph_name = "graph_" + str(i)
                sess_name = "sess_" + str(i)
                graph_name, bottleneck_tensor, resized_image_tensor = (
                    create_model_graph(pretrained_model_info)
                )

                with tf.Session(graph=graph_name) as sess_name:
                    jpeg_data_tensor, decoded_image_tensor = add_jpeg_decoding(
                        pretrained_model_info['input_width'], pretrained_model_info['input_height'],
                        pretrained_model_info['input_depth'], pretrained_model_info['input_mean'],
                        pretrained_model_info['input_std'])

                    cache_bottlenecks(sess_name, image_lists, GENERAL_SETTING['image_dir'],
                                      GENERAL_SETTING['bottleneck_dir'], jpeg_data_tensor,
                                      decoded_image_tensor, resized_image_tensor,
                                      bottleneck_tensor, pretrained_model_info['architecture'])
                    print("cached botteneck!, sess: ", sess_name)
                    sess_name.close()

            base_testing_accuracy, base_last_layer_train_results, base_last_layer_test_results = training_operation(
                image_lists=image_lists,
                pretrained_model_infos=pretrained_model_infos,
                MODEL_SETTING=MODEL_SETTING,
                training_fold_names=training_fold_names,
                testing_fold_names=testing_fold_names)

            MODEL_SETTING['test_accuracies'].append(base_testing_accuracy)

            testing_accuracy.append(base_testing_accuracy)
            all_last_layer_train_results.append(base_last_layer_train_results)
            all_last_layer_test_results.append(base_last_layer_test_results)

            # skip the 3 feature extractor model
            if (model_index >= 1):
                three_bases_last_layer_train_results.append(base_last_layer_train_results)
                three_bases_last_layer_test_results.append(base_last_layer_test_results)

        ########    Ensemble training   #########

        # Three bases models
        three_bases_naive_averaging_with_logit_test_accurary, _ = naive_averaging_result(image_lists= image_lists, class_count=class_count,
                                                                                      all_last_layer_test_results=three_bases_last_layer_test_results,
                                                                                      is_logit=True)
        three_bases_naive_averaging_with_softmax_test_accurary, _= naive_averaging_result(image_lists= image_lists, class_count=class_count,
                                                                                        all_last_layer_test_results=three_bases_last_layer_test_results,
                                                                                        is_logit=False)
        three_bases_naive_ensemble_logits_test_accuracies.append(three_bases_naive_averaging_with_logit_test_accurary)
        three_bases_naive_ensemble_softmax_test_accuracies.append(
            three_bases_naive_averaging_with_softmax_test_accurary)

        # All models
        all_naive_averaging_with_logit_test_accurary, all_naive_averaging_with_logit_conf = naive_averaging_result(image_lists= image_lists, class_count=class_count,
                                                                              all_last_layer_test_results=all_last_layer_test_results,
                                                                              is_logit=True)
        all_naive_averaging_with_softmax_test_accurary, all_naive_averaging_with_softmax_conf = naive_averaging_result(image_lists= image_lists, class_count=class_count,
                                                                                all_last_layer_test_results=all_last_layer_test_results,
                                                                                is_logit=False)
        all_naive_ensemble_logits_test_accuracies.append(all_naive_averaging_with_logit_test_accurary)
        all_naive_ensemble_softmax_test_accuracies.append(all_naive_averaging_with_softmax_test_accurary)
        all_naive_averaging_with_logit_conf_arr.append(all_naive_averaging_with_logit_conf)
        all_naive_averaging_with_softmax_conf_arr.append(all_naive_averaging_with_softmax_conf)

        # Print and save to file
        print("base learner testing accuracy", testing_accuracy)
        print("naive_averaging_with_logit_test_accurary", all_naive_averaging_with_logit_test_accurary)
        print("naive_averaging_with_softmax_test_accurary", all_naive_averaging_with_softmax_test_accurary)

        base_learner_test_results = ['', datetime.now(), '', '', '', '',
                                     'base_learner_test_accuracy', testing_accuracy]
        save_to_csv(GENERAL_SETTING['csvlogfile'], [base_learner_test_results])

        # Three bases
        three_bases_naive_averaging_with_logit_result = ['', datetime.now(), '', '', '', '',
                                                         'three_bases_naive_averaging_with_logit_test_accuracy',
                                                         three_bases_naive_averaging_with_logit_test_accurary * 100]
        save_to_csv(GENERAL_SETTING['csvlogfile'], [three_bases_naive_averaging_with_logit_result])

        three_bases_naive_averaging_with_softmax_result = ['', datetime.now(), '', '', '', '',
                                                           'three_bases_naive_averaging_with_softmax_test_accuracy',
                                                           three_bases_naive_averaging_with_softmax_test_accurary * 100]
        save_to_csv(GENERAL_SETTING['csvlogfile'], [three_bases_naive_averaging_with_softmax_result])

        # All
        all_naive_averaging_with_logit_result = ['', datetime.now(), '', '', '', '',
                                                 'all_naive_averaging_with_logit_test_accuracy',
                                                 all_naive_averaging_with_logit_test_accurary * 100, all_naive_averaging_with_logit_conf]
        save_to_csv(GENERAL_SETTING['csvlogfile'], [all_naive_averaging_with_logit_result])

        all_naive_averaging_with_softmax_result = ['', datetime.now(), '', '', '', '',
                                                   'all_naive_averaging_with_softmax_test_accuracy',
                                                   all_naive_averaging_with_softmax_test_accurary * 100, all_naive_averaging_with_softmax_conf]
        save_to_csv(GENERAL_SETTING['csvlogfile'], [all_naive_averaging_with_softmax_result])

    # Calculate Mean and stddev of performance over 30 testing times. Save to file
    save_mean_and_std_result(three_bases_naive_ensemble_softmax_test_accuracies, 'three bases ensemble softmax')
    save_mean_and_std_result(three_bases_naive_ensemble_logits_test_accuracies, 'three bases ensemble logit')

    save_mean_and_std_result(all_naive_ensemble_softmax_test_accuracies, 'all ensemble softmax')
    save_avg_conf(all_naive_averaging_with_softmax_conf_arr, 'all_ensemble_softmax', image_lists)
    save_mean_and_std_result(all_naive_ensemble_logits_test_accuracies, 'all ensemble logit')
    save_avg_conf(all_naive_averaging_with_logit_conf_arr, 'all_ensemble_logit', image_lists)

    for MODEL_SETTING in MODEL_SETTINGS:
        save_mean_and_std_result(MODEL_SETTING['test_accuracies'], str(MODEL_SETTING['architecture']))


if __name__ == '__main__':
    tf.app.run(main=main)
