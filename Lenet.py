import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import flatten
from tensorflow.examples.tutorials.mnist import input_data

# Hyperparameters
mu = 0
sigma = 0.1
layer_depth = {
    'layer_1': 6,
    'layer_2': 16,
    'layer_3': 120,
    'layer_f1': 84
}
learning_rate = 0.001
EPOCHS = 10
BATCH_SIZE = 128

def cnn_model_fn(features, labels, mode):

    #[batch_size, width, height, channel]
    #batch_size = -1 for dynamic batch size; channel =1 for BW image
    input_layer = tf.reshape(features, [-1,28,28, 1])

    #padding = same - the output tensor has same width and height with the input tensor
    #filter = 32: 32 filters to apply
    #output tensor size: [batch_size, 28,28, 32]
    conv1 = tf.layers.conv2d(
        inputs = input_layer,
        filters=32,
        kernel_size = [5,5],
        padding = "same",
        activation=tf.nn.relu
    )

    pool1 = tf.layers.max_pooling2d(inputs = conv1, pool_size =[2,2], stride =2 )

def LeNet(x):
    # TODO: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    conv1_w = tf.Variable(tf.truncated_normal(shape=[5, 5, 1, 6], mean=mu, stddev=sigma))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1 = tf.nn.conv2d(x, conv1_w, strides=[1, 1, 1, 1], padding='VALID') + conv1_b
    # TODO: Activation.
    conv1 = tf.nn.relu(conv1)

    # TODO: Pooling. Input 28x28x6. Output 14x14X6
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # TODO: Layer 2: Conv. Output: 10x10x16
    conv2_w = tf.Variable(tf.truncated_normal(shape=[5, 5, 6, 16], mean=mu, stddev=sigma))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2 = tf.nn.conv2d (pool1, conv2_w, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
    # TODO: Activation
    conv2 = tf.nn.relu(conv2)

    # TODO: Pooling
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    # TODO: Flatten
    fc1 = flatten(pool2)


    # TODO: Layer 3: Fully connected
    fc1_w = tf.Variable(tf.truncated_normal(shape=(400,120), mean=mu, stddev=sigma))
    fc1_b = tf.Variable(tf.zeros(120))
    fc1 = tf.matmul(fc1, fc1_w) + fc1_b

    # TODO: Activation
    fc1 = tf.nn.relu(fc1)

    # TODO: Layer 4: Fully connected. Input 120, output 84
    fc2_w = tf.Variable(tf.truncated_normal(shape=(120, 84), mean=mu, stddev=sigma))
    fc2_b = tf.Variable(tf.zeros(84))
    fc2 = tf.matmul(fc1, fc2_w) + fc2_b
    # TODO: Activation
    fc2 = tf.nn.relu(fc2)


    # TODO: Layer 5: Fully connected. Input 84, output 10
    fc3_w = tf.Variable(tf.truncated_normal(shape=(84, 10), mean=mu, stddev = sigma))
    fc3_b = tf.Variable(tf.zeros(10))
    logits = tf.matmul(fc2, fc3_w) + fc3_b
    return logits


def main(unused_argv):

    mnist = input_data.read_data_sets("MNIST_data/", reshape=False)
    X_train, y_train = mnist.train.images, mnist.train.labels
    X_validation, y_validation = mnist.validation.images, mnist.validation.labels
    X_test, y_test = mnist.test.images, mnist.test.labels

    assert (len(X_train) == len(y_train))
    assert (len(X_validation) == len(y_validation))
    assert (len(X_test) == len(y_test))

    print()
    print("Image Shape: {}".format(X_train[0].shape))
    print()
    print("Training Set:   {} samples".format(len(X_train)))
    print("Validation Set: {} samples".format(len(X_validation)))
    print("Test Set:       {} samples".format(len(X_test)))
    x = tf.placeholder(tf.float32, (None, 32, 32, 1))
    y = tf.placeholder(tf.int32, (None))
    one_hot_y = tf.one_hot(y, 10)

    # TODO: resize the 28x28x1 of MNIST to 32x32x1 of LeNet
    # Pad images with 0s with two rows of zeros on the top and bottom,
    # and two columns of zeros on the left and right (28+2+2 = 32).
    X_train = np.pad(X_train, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
    X_validation = np.pad(X_validation, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
    X_test = np.pad(X_test, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')

    print("Updated Image Shape: {}".format(X_train[0].shape))

    ###########################################################################
    logits = LeNet(x)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_y)
    loss_op = tf.reduce_mean(cross_entropy)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    training_op = optimizer.minimize(loss_op)

    correct_prediction = tf.equal(tf.arg_max(logits, 1), tf.argmax(one_hot_y, 1))
    accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    saver = tf.train.Saver()

    def evaluate(X_data, y_data):
        num_examples = len(X_data)
        total_accuracy = 0
        sess = tf.get_default_session()
        for offset in range(0, num_examples, BATCH_SIZE):
            batch_x, batch_y = X_data[offset: offset + BATCH_SIZE], y_data[offset: offset + BATCH_SIZE]
            accuracy = sess.run(accuracy_op, feed_dict={x: batch_x, y: batch_y})
            total_accuracy += (accuracy*len(batch_x))
        return total_accuracy/num_examples

    # TODO: Train
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        num_examples = len(X_train)
        print("Training...")
        print()
        for i in range(EPOCHS):

            for offset in range(0, num_examples, BATCH_SIZE):
                end = offset + BATCH_SIZE
                batch_x, batch_y = X_train[offset:end], y_train[offset:end]
                sess.run(training_op, feed_dict={x: batch_x, y: batch_y})

            validation_accuracy = evaluate(X_validation, y_validation)
            print("EPOCH {} ...".format(i + 1))
            print("Validation Accuracy = {:.3f}".format(validation_accuracy))
            print()

        saver.save(sess, 'lenet')
        print("Model saved")

    # TODO: Test
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint('.'))

        test_accuracy = evaluate(X_test, y_test)
        print("Test Accuracy = {:.3f}".format(test_accuracy))

if __name__ == '__main__':
    tf.app.run()
