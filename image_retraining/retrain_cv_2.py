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

FLAGS = None

############## Provide the model that you want to train here ##############

# ARCHITECTURES = ['inception_v3', 'mobilenet_1.0_224', 'resnet_v2']
# ARCHITECTURES = ['resnet_v2']
ARCHITECTURES = ['inception_resnet_v2']
#ARCHITECTURES = ['resnet50']


# These are all parameters that are tied to the particular model architecture
# we're using for Inception v3. These include things like tensor names and their
# sizes. If you want to adapt this script to work with another model, you will
# need to update these to reflect the values in the network you're using.
MAX_NUM_IMAGES_PER_CLASS = 2 ** 27 - 1  # ~134M


def save_to_csv(filename, data_arr):
    f = open(filename, 'a')
    with f:
        writer = csv.writer(f)
        for row in data_arr:
            writer.writerow(row)


def create_image_lists(image_dir, testing_percentage):
    """Builds a list of training images from the file system.

    Analyzes the sub folders in the image directory, splits them into stable
    training, testing, and validation sets, and returns a data structure
    describing the lists of images for each label and their paths.

    Args:
      image_dir: String path to a folder containing subfolders of images.
      testing_percentage: Integer percentage of the images to reserve for tests.
      validation_percentage: Integer percentage of images reserved for validation.

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
        # training_images = []
        testing_images = []
        # validation_images = []
        fold1 = []
        fold2 = []
        fold3 = []
        fold4 = []


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
            if percentage_hash < testing_percentage:
                testing_images.append(base_name)
            elif percentage_hash < (testing_percentage + 20):
                fold1.append(base_name)
            elif percentage_hash < (testing_percentage + 40):
                fold2.append(base_name)
            elif percentage_hash < (testing_percentage + 60):
                fold3.append(base_name)
            else:
                fold4.append(base_name)

        result[label_name] = {
            'dir': dir_name,
            'testing': testing_images,
            'fold1': fold1,
            'fold2': fold2,
            'fold3': fold3,
            'fold4': fold4
        }
    print("cross validation folds and test images", result)
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

    if category not in label_lists:
        tf.logging.fatal('Category does not exist %s.', category)
    category_list = label_lists[category]
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
        model_path = os.path.join(FLAGS.model_dir, model_info['model_file_name'])
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


bottleneck_path_2_bottleneck_values = {}


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
        for category in ['testing', 'fold1', 'fold2', 'fold3', 'fold4']:
            category_list = label_lists[category]
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

                architecture = ARCHITECTURES[i]
                bottleneck = get_bottleneck(
                    image_lists, label_name, image_index, image_dir, category,
                    bottleneck_dir, architecture)
                merged_bottleneck.append(bottleneck)
                merged_bottleneck_shape += len(bottleneck)

                ground_truth = np.zeros(class_count, dtype=np.float32)
                ground_truth[label_index] = 1.0
                if (i == len(ARCHITECTURES) - 1):
                    ground_truths.append(ground_truth)
                    filenames.append(image_name)

            merged_bottleneck = np.reshape(merged_bottleneck, merged_bottleneck_shape)

            bottlenecks.append(merged_bottleneck)

    else:
        # Retrieve all bottlenecks.
        for label_index, label_name in enumerate(image_lists.keys()):
            # choose random fold from list of fold in training set
            category = random.choice(categories)
            for image_index, image_name in enumerate(
                    image_lists[label_name][category]):
                image_name = get_image_path(image_lists, label_name, image_index,
                                            image_dir, category)
                merged_bottleneck = []
                merged_bottleneck_shape = 0
                for i in range(len(ARCHITECTURES)):

                    architecture = ARCHITECTURES[i]
                    bottleneck = get_bottleneck(
                        image_lists, label_name, image_index, image_dir, category,
                        bottleneck_dir, architecture)
                    merged_bottleneck.append(bottleneck)
                    merged_bottleneck_shape += len(bottleneck)

                    ground_truth = np.zeros(class_count, dtype=np.float32)
                    ground_truth[label_index] = 1.0
                    if (i == len(ARCHITECTURES) - 1):
                        ground_truths.append(ground_truth)
                        filenames.append(image_name)

                merged_bottleneck = np.reshape(merged_bottleneck, merged_bottleneck_shape)

                bottlenecks.append(merged_bottleneck)
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
        sess, graph.as_graph_def(), [FLAGS.final_tensor_name])
    with gfile.FastGFile(graph_file_name, 'wb') as f:
        f.write(output_graph_def.SerializeToString())
    return


def prepare_file_system():
    # Setup the directory we'll write summaries to for TensorBoard
    if tf.gfile.Exists(FLAGS.summaries_dir):
        tf.gfile.DeleteRecursively(FLAGS.summaries_dir)
    tf.gfile.MakeDirs(FLAGS.summaries_dir)
    if FLAGS.intermediate_store_frequency > 0:
        ensure_dir_exists(FLAGS.intermediate_output_graphs_dir)
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


    # elif architecture == 'resnet_v2':
    #     bottleneck_tensor_name = 'resnet_v2_152/pool5:0'
    #     bottleneck_tensor_size = 2048
    #     input_width = 230
    #     input_height = 230
    #     input_depth = 3
    #     resized_input_tensor_name = 'resnet_v2_152/Pad:0'
    #     model_file_name = 'resnet_output.pb'
    #     input_mean = 128
    #     input_std = 128
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
        input_height =224
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


def add_final_training_ops(class_count, final_tensor_name, bottleneck_tensor_size, hidden_layer1_size):
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
        adaptive_learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, global_step, 1000, FLAGS.learning_rate_decay, staircase=True)

        #optimizer = tf.train.GradientDescentOptimizer(FLAGS.learning_rate)
        optimizer = tf.train.GradientDescentOptimizer(adaptive_learning_rate)
        train_step = optimizer.minimize(cross_entropy_mean)

    return (train_step, cross_entropy_mean, bottleneck_input, ground_truth_input,
            final_tensor, keep_prob, global_step)


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
    # Early stopping
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

def early_stopping_by_loss (train_cross_entropy_values, early_stopping_n_steps):
    n_last_losses =[]
    if len(train_cross_entropy_values)>early_stopping_n_steps:
        for i in range(len(train_cross_entropy_values)-early_stopping_n_steps, len (train_cross_entropy_values)):
            result = train_cross_entropy_values[i]
            n_last_losses.append(result)

        ascending_n_last_losses = sorted(n_last_losses, key=itemgetter('loss'), reverse=False)
        if (n_last_losses[0]['loss'] == ascending_n_last_losses[0]['loss']):
            return False, n_last_losses[0]
        else:
            return True, n_last_losses[0]
    else:
        return True, train_cross_entropy_values[0]

def get_checkpoint_path(step, training_fold_names):
    fold_indexs=""
    for training_fold_name in training_fold_names:
        fold_indexs= fold_indexs+training_fold_name[4:len(training_fold_name)] #foldi-> i

    model = ''
    for architecture in ARCHITECTURES:
        model = model+"_"+architecture

    checkpoint_name = "model_" +str(model)+ "_step_" + str(step) + "_folds_"+str(fold_indexs)+  "_lr_" + str(FLAGS.learning_rate) + "_test_p_" + str(
        FLAGS.testing_percentage) + \
                      "_val_p_" + str(FLAGS.validation_percentage) + "_eval_inv_" + str(FLAGS.eval_step_interval) + \
                      "_train_b_" + str(FLAGS.train_batch_size) + "_hidden1_" + str(FLAGS.hidden_layer1_size) + \
                      "_dropout_" + str(FLAGS.dropout_keep_prob) + "_early_s" + str(FLAGS.early_stopping_n_steps)
    checkpoint_path = os.path.join(FLAGS.checkpoint_dir, checkpoint_name)
    return checkpoint_path + '.cpkt'


def delete_and_update_checkpoint_arr(checkpoint_path_arr):
    num_to_keep = FLAGS.early_stopping_n_steps
    if (len(checkpoint_path_arr) > num_to_keep):
        checkpoint_to_del = checkpoint_path_arr.pop(0)
        try:
            os.remove(checkpoint_to_del)
        except:
            tf.logging.info("Checkpoint dir to del does not exists")
    return checkpoint_path_arr

def delete_all_checkpoints(checkpoint_path_arr):
    for checkpoint_path in checkpoint_path_arr:
        checkpoint_path_metafile = checkpoint_path+".meta"
        try:
            os.remove(checkpoint_path)
            os.remove(checkpoint_path_metafile)
        except:
            tf.logging.info("Checkpointdir to del does not exists")

def train_validation_two_decimal(train_accuracy, validation_accuracy):
    train_accuracy_two_decimal = int(train_accuracy * 10000) / 100
    validation_accuracy_two_decimal = int(validation_accuracy * 10000) / 100
    return train_accuracy_two_decimal, validation_accuracy_two_decimal

def training_operation(image_lists, model_infos, is_final_train, training_fold_names, validation_fold_name):

    print("train folds", training_fold_names, "validation fold", validation_fold_name)

    train_validation_loss_results =[]
    train_cross_entropy_values = []

    checkpoint_path_arr = []

    tf.reset_default_graph()
    with tf.Session() as sess:
        bottleneck_tensor_size = 0
        for model_info in model_infos:
            bottleneck_tensor_size += model_info['bottleneck_tensor_size']

        (train_step, cross_entropy, bottleneck_input, ground_truth_input,
         final_tensor, keep_prob, global_step) = add_final_training_ops(
            len(image_lists.keys()), FLAGS.final_tensor_name, bottleneck_tensor_size,
            FLAGS.hidden_layer1_size)

        # Create the operations we need to evaluate the accuracy of our new layer.
        evaluation_step, prediction = add_evaluation_step(
            final_tensor, ground_truth_input)

        # Merge all the summaries and write them out to the summaries_dir
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train',
                                             sess.graph)

        validation_writer = tf.summary.FileWriter(
            FLAGS.summaries_dir + '/validation')

        # Set up all our weights to their initial default values.
        init = tf.global_variables_initializer()
        sess.run(init)


        is_continue_training = True

        for i in range(FLAGS.how_many_training_steps):

            if (is_continue_training):
                (train_bottlenecks, train_ground_truth, _) = get_random_cached_bottlenecks(
                    image_lists, FLAGS.train_batch_size, training_fold_names,
                    FLAGS.image_dir, FLAGS.bottleneck_dir, ARCHITECTURES)
                # print (i, "-", train_bottlenecks)

                train_summary, _ = sess.run(
                    [merged, train_step],
                    feed_dict={bottleneck_input: train_bottlenecks,
                               ground_truth_input: train_ground_truth,
                               keep_prob: FLAGS.dropout_keep_prob,
                               global_step: i})
                if is_final_train:
                    train_writer.add_summary(train_summary, i)

                is_last_step = (i + 1 == FLAGS.how_many_training_steps)
                if (i % FLAGS.eval_step_interval) == 0 or is_last_step:
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

                    if is_final_train:
                        train_accuracy_two_decimal, validation_accuracy_two_decimal = train_validation_two_decimal(
                            train_accuracy, 0)
                        intermediate_result = ['', datetime.now(), i, train_accuracy * 100,cross_entropy_value,'',
                                               training_fold_names]

                    else:
                        validation_bottlenecks, validation_ground_truth, _ = (
                            get_random_cached_bottlenecks(image_lists, FLAGS.validation_batch_size,
                                                          validation_fold_name, FLAGS.image_dir, FLAGS.bottleneck_dir,
                                                          ARCHITECTURES)
                        )
                        validation_summary, validation_accuracy = sess.run(
                            [merged, evaluation_step],
                            feed_dict={bottleneck_input: validation_bottlenecks,
                                       ground_truth_input: validation_ground_truth, keep_prob: 1.0})

                        tf.logging.info('%s: Step %d: Validation accuracy = %.1f%% (N=%d)' %
                                        (datetime.now(), i, validation_accuracy * 100,
                                         len(validation_bottlenecks)))
                        train_accuracy_two_decimal, validation_accuracy_two_decimal = train_validation_two_decimal(
                            train_accuracy,
                            validation_accuracy)
                        intermediate_result = ['', datetime.now(), i, train_accuracy * 100, cross_entropy_value, validation_accuracy * 100,
                                           training_fold_names]

                    # Print the result to csvlogfile

                    save_to_csv(FLAGS.csvlogfile, [intermediate_result])


                    if i >= 1000:
                        # Save a checkpoint
                        checkpoint_path = get_checkpoint_path(i, training_fold_names)
                        checkpoint_path_arr.append(checkpoint_path)

                        tf.train.Saver(write_version=tf.train.SaverDef.V1).save(sess, checkpoint_path)
                        checkpoint_path_arr = delete_and_update_checkpoint_arr(checkpoint_path_arr)
                        # print("checkpoint_path_arr", checkpoint_path_arr)

                        if is_final_train:
                            train_cross_entropy_value = {
                                'step': i,
                                'train_accuracy': train_accuracy_two_decimal,
                                'loss': cross_entropy_value,
                                'checkpoint_path': checkpoint_path
                            }
                            train_cross_entropy_values.append(train_cross_entropy_value)

                            #Early stopping condition check
                            is_continue_training, result = early_stopping_by_loss(train_cross_entropy_values,
                                                                                  FLAGS.early_stopping_n_steps)
                            if not is_continue_training:
                                tf.logging.info("Early stopping. The best result is at %d steps: Train accuracy %.1f%%,"
                                                 "Loss: %.f", result['step'],
                                                result['train_accuracy'],
                                                result['loss'])

                                early_stopping_logging_info = "The best final result is at " + str(
                                    result['step']) + " steps:" + \
                                                              "Train accuracy: " + str(
                                    result['train_accuracy']) + ", Loss: " + str(result['loss'])

                                early_stopping_result = ['', '', '', '', '','', '', '', '', early_stopping_logging_info]
                                save_to_csv(FLAGS.csvlogfile, [early_stopping_result])
                        else:
                            # Store the result into array
                            train_validation_loss_result = {
                                'step': i,
                                'train_accuracy': train_accuracy_two_decimal,
                                'validation_accuracy': validation_accuracy_two_decimal,
                                'loss': cross_entropy_value,
                                'checkpoint_path': checkpoint_path
                            }
                            train_validation_loss_results.append(train_validation_loss_result)

                            # Early stopping condition check
                            is_continue_training, result = early_stopping_by_validation(train_validation_loss_results,
                                                                                        FLAGS.early_stopping_n_steps)
                            if not is_continue_training:
                                tf.logging.info("Early stopping. The best result is at %d steps: Train accuracy %.1f%%,"
                                                " Validation accuracy %.1f%%, Loss: %.f", result['step'],
                                                result['train_accuracy'],
                                                result['validation_accuracy'], result['loss'])

                                early_stopping_logging_info = "The best result is at " + str(
                                    result['step']) + " steps:" + \
                                                              "Train accuracy: " + str(
                                    result['train_accuracy']) + ", Validation accuracy: " + \
                                                              str(result['validation_accuracy']) + ", Loss: " + str(
                                    result['loss'])

                                early_stopping_result = ['', '', '', '', '', '','', '', '', early_stopping_logging_info]
                                save_to_csv(FLAGS.csvlogfile, [early_stopping_result])

            else:
                break

        # We've completed all our training, so run a final test evaluation on
        # some new images we haven't used before.
        test_bottlenecks, test_ground_truth, test_filenames = (
            get_random_cached_bottlenecks(
                image_lists, FLAGS.test_batch_size, ['testing'],
                FLAGS.image_dir, FLAGS.bottleneck_dir, ARCHITECTURES))

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

        if FLAGS.print_misclassified_test_images:

            tf.logging.info('=== MISCLASSIFIED TEST IMAGES ===')
            for i, test_filename in enumerate(test_filenames):
                if predictions[i] != test_ground_truth[i].argmax():
                    tf.logging.info('%s  %s' %
                                    (test_filename,
                                     list(image_lists.keys())[predictions[i]]))
                    misclassified_image_arr.append((test_filename,
                                                    list(image_lists.keys())[predictions[i]]))

        # Print the result to csvlogfile
        final_result = ['', datetime.now(), '', '', '','',
                        '', test_accuracy * 100, FLAGS.summaries_dir, misclassified_image_arr]
        save_to_csv(FLAGS.csvlogfile, [final_result])

        if not is_final_train:
            print ("checkpoints will be deleted", checkpoint_path_arr)
            delete_all_checkpoints(checkpoint_path_arr)

    sess.close()
    if not is_final_train:
        return result['validation_accuracy']
    else:
        return

def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    prepare_file_system()

    model_infos = []
    for architecture in ARCHITECTURES:
        model_info = create_model_info(architecture)
        model_infos.append(model_info)

        # Look at the folder structure, and create lists of all the images.
    image_lists = create_image_lists(FLAGS.image_dir, FLAGS.testing_percentage)
    class_count = len(image_lists.keys())
    if class_count == 0:
        tf.logging.error('No valid folders of images found at ' + FLAGS.image_dir)
        return -1
    if class_count == 1:
        tf.logging.error('Only one valid folder of images found at ' +
                         FLAGS.image_dir +
                         ' - multiple classes are needed for classification.')
        return -1

    graph_infos = []
    for i in range(len(model_infos)):
        model_info = model_infos[i]
        print("model_infos", model_infos, "model_info", model_info)
        graph_name = "graph_" + str(i)
        sess_name = "sess_" + str(i)
        graph_name, bottleneck_tensor, resized_image_tensor = (
            create_model_graph(model_info)
        )
        graph_infos.append({'graph': graph_name,
                            'bottleneck_tensor': bottleneck_tensor,
                            'resized_image_tensor': resized_image_tensor
                            })

        with tf.Session(graph=graph_name) as sess_name:
            jpeg_data_tensor, decoded_image_tensor = add_jpeg_decoding(
                model_info['input_width'], model_info['input_height'],
                model_info['input_depth'], model_info['input_mean'],
                model_info['input_std'])

            cache_bottlenecks(sess_name, image_lists, FLAGS.image_dir,
                              FLAGS.bottleneck_dir, jpeg_data_tensor,
                              decoded_image_tensor, resized_image_tensor,
                              bottleneck_tensor, model_info['architecture'])
            print("cached botteneck!, sess: ", sess_name)
            # init = tf.global_variables_initializer()
            sess_name.close()

    # Print the FLAGS setting to logfile.csv
    settings = [FLAGS, ARCHITECTURES]
    save_to_csv(FLAGS.csvlogfile, [[settings]])

    # Cross validation training
    cross_validation_accuracy_arr=[]
    for validation_fold_index in range(1, 5): #fold 1,2,3,4
        validation_fold_name = ["fold" + str(validation_fold_index)]

        training_fold_names = []
        for training_fold_index in [v for v in range(1, 5) if v != validation_fold_index]:
            training_fold_name = "fold" + str(training_fold_index)
            training_fold_names.append(training_fold_name)


        cross_validation_accuracy = training_operation(image_lists=image_lists, model_infos = model_infos, is_final_train=False,
                           training_fold_names=training_fold_names, validation_fold_name=validation_fold_name)
        cross_validation_accuracy_arr.append(cross_validation_accuracy)

    # Final training and validation:
    training_fold_names = ['fold1', 'fold2', 'fold3', 'fold4']
    validation_fold_name = []
    training_operation(image_lists=image_lists, model_infos = model_infos, is_final_train=True,
                           training_fold_names=training_fold_names, validation_fold_name=validation_fold_name)
    average_cross_validation_accuracy = sum(cross_validation_accuracy_arr)/float(len(cross_validation_accuracy_arr))
    cross_validation_result= ['', '', '', '', '',average_cross_validation_accuracy]
    save_to_csv(FLAGS.csvlogfile, [cross_validation_result])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--image_dir',
        type=str,
        default='',
        help='Path to folders of labeled images.'
    )
    parser.add_argument(
        '--output_graph',
        type=str,
        default='/tmp/output_graph.pb',
        help='Where to save the trained graph.'
    )
    parser.add_argument(
        '--intermediate_output_graphs_dir',
        type=str,
        default='/tmp/intermediate_graph/',
        help='Where to save the intermediate graphs.'
    )
    parser.add_argument(
        '--intermediate_store_frequency',
        type=int,
        default=0,
        help="""\
         How many steps to store intermediate graph. If "0" then will not
         store.\
      """
    )
    parser.add_argument(
        '--output_labels',
        type=str,
        default='/tmp/output_labels.txt',
        help='Where to save the trained graph\'s labels.'
    )
    parser.add_argument(
        '--summaries_dir',
        type=str,
        default='/tmp/retrain_logs/BreastCancer_1808',
        help='Where to save summary logs for TensorBoard.'
    )
    parser.add_argument(
        '--how_many_training_steps',
        type=int,
        default=4000,
        help='How many training steps to run before ending.'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.01,
        help='How large a learning rate to use when training.'
    )
    parser.add_argument(
        '--testing_percentage',
        type=int,
        default=10,
        help='What percentage of images to use as a test set.'
    )
    parser.add_argument(
        '--validation_percentage',
        type=int,
        default=10,
        help='What percentage of images to use as a validation set.'
    )
    parser.add_argument(
        '--eval_step_interval',
        type=int,
        default=100,
        help='How often to evaluate the training results.'
    )
    parser.add_argument(
        '--train_batch_size',
        type=int,
        default=100,
        help='How many images to train on at a time.'
    )
    parser.add_argument(
        '--test_batch_size',
        type=int,
        default=-1,
        help="""\
      How many images to test on. This test set is only used once, to evaluate
      the final accuracy of the model after training completes.
      A value of -1 causes the entire test set to be used, which leads to more
      stable results across runs.\
      """
    )
    parser.add_argument(
        '--validation_batch_size',
        type=int,
        default=-1,
        help="""\
      How many images to use in an evaluation batch. This validation set is
      used much more often than the test set, and is an early indicator of how
      accurate the model is during training.
      A value of -1 causes the entire validation set to be used, which leads to
      more stable results across training iterations, but may be slower on large
      training sets.\
      """
    )
    parser.add_argument(
        '--hidden_layer1_size',
        type=int,
        default=50,
        help="""\
    Specify number of neurons in the fully connected hidden layer 1.\
    """
    )
    parser.add_argument(
        '--dropout_keep_prob',
        type=np.float32,
        default=1.0,
        help="""\
    Specify the probality to keep neurons in dropout. \
    """
    )
    parser.add_argument(
        '--print_misclassified_test_images',
        default=True,
        help="""\
      Whether to print out a list of all misclassified test images.\
      """,
        action='store_true'
    )
    parser.add_argument(
        '--model_dir',
        type=str,
        default='/home/duclong002/pretrained_model/',
        help="""\
      Path to classify_image_graph_def.pb,
      imagenet_synset_to_human_label_map.txt, and
      imagenet_2012_challenge_label_map_proto.pbtxt.\
      """
    )
    parser.add_argument(
        '--bottleneck_dir',
        type=str,
        default='/tmp/bottleneck',
        help='Path to cache bottleneck layer values as files.'
    )
    parser.add_argument(
        '--final_tensor_name',
        type=str,
        default='final_result',
        help="""\
      The name of the output classification layer in the retrained graph.\
      """
    )
    parser.add_argument(
        '--flip_left_right',
        default=False,
        help="""\
      Whether to randomly flip half of the training images horizontally.\
      """,
        action='store_true'
    )
    parser.add_argument(
        '--random_crop',
        type=int,
        default=0,
        help="""\
      A percentage determining how much of a margin to randomly crop off the
      training images.\
      """
    )
    parser.add_argument(
        '--random_scale',
        type=int,
        default=0,
        help="""\
      A percentage determining how much to randomly scale up the size of the
      training images by.\
      """
    )
    parser.add_argument(
        '--random_brightness',
        type=int,
        default=0,
        help="""\
      A percentage determining how much to randomly multiply the training image
      input pixels up or down by.\
      """
    )
    parser.add_argument(
        '--architectures',
        type=str,
        default='',
        help="""\
      Which model architecture to use. 'inception_v3' is the most accurate, but
      also the slowest. For faster or smaller models, chose a MobileNet with the
      form 'mobilenet_<parameter size>_<input_size>[_quantized]'. For example,
      'mobilenet_1.0_224' will pick a model that is 17 MB in size and takes 224
      pixel input images, while 'mobilenet_0.25_128_quantized' will choose a much
      less accurate, but smaller and faster network that's 920 KB on disk and
      takes 128x128 images. See https://research.googleblog.com/2017/06/mobilenets-open-source-models-for.html
      for more information on Mobilenet.\
      """)

    parser.add_argument(
        '--csvlogfile',
        type=str,
        default='',
        help='Link to logfile.csv'
    )

    parser.add_argument(
        '--early_stopping_n_steps',
        type=int,
        default=10,
        help='Number of further validation steps to be executed before early stopping'
    )
    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        default="/home/duclong002/checkpoints",
        help="Checkpoint dir"

    )
    parser.add_argument(
        '--learning_rate_decay',
        type=float,
        default=0.8,
        help="How much learning rate be decayed after 100 steps (=0.8 -> 1000 steps learning rate = 10%)"
    )

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

