from datetime import datetime
from functools import partial

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def log_dir(prefix=""):
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    root_log_dir = "cnn_model/tb"
    if prefix:
        prefix += "-"
    name = prefix + "run-" + now
    return "{}/{}/".format(root_log_dir, name)


height = 28
width = 28
channels = 1  # Greyscale
n_inputs = height * width

conv1_feature_maps = 32
conv1_kernel_size = 3
conv1_stride = 1
conv1_pad = "SAME"

conv2_feature_maps = 64
conv2_kernel_size = 3
conv2_stride = 2
conv2_pad = "SAME"

pool3_feature_maps = conv2_feature_maps

n_neurons_fc1 = 64
n_outputs = 10

with tf.name_scope('Input'):
    X = tf.placeholder(tf.float32, shape=(None, n_inputs))
    X_ = tf.reshape(X, shape=(-1, height, width, channels), name='X_')
    y = tf.placeholder(tf.int32, shape=None, name='y')

conv1 = tf.layers.conv2d(X_,
                         filters=conv1_feature_maps,
                         kernel_size=conv1_kernel_size,
                         strides=conv1_stride,
                         padding=conv1_pad,
                         activation=tf.nn.relu,
                         name='Conv1'
                         )

conv2 = tf.layers.conv2d(conv1,
                         filters=conv2_feature_maps,
                         kernel_size=conv2_kernel_size,
                         strides=conv2_stride,
                         padding=conv1_pad,
                         activation=tf.nn.relu,
                         name='Conv2'
                         )

with tf.name_scope('Pool3'):
    pool3 = tf.reshape(
        tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID'),
        shape=[-1, pool3_feature_maps * 7 * 7]
    )

with tf.name_scope('FullyConnected'):
    training = tf.placeholder_with_default(True, shape=(), name='training')
    bn_layer = partial(tf.layers.batch_normalization, training=training, momentum=0.9)
    fully_connected = tf.nn.relu(bn_layer(
        tf.layers.dense(pool3, n_neurons_fc1, kernel_initializer=tf.variance_scaling_initializer())))

with tf.name_scope('Output'):
    logits = bn_layer(tf.layers.dense(fully_connected, n_outputs, name='output'))

with tf.name_scope('Train'):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y)
    loss = tf.reduce_mean(cross_entropy)
    training_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
    loss_summary = tf.summary.scalar('log_loss', loss)

with tf.name_scope('Evaluate'):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    accuracy_summary = tf.summary.scalar('accuracy', accuracy)

# Now that the architecture is set, let's read the data
mnist = input_data.read_data_sets('/tmp/data')
file_writer = tf.summary.FileWriter(log_dir('mnist_cnn'), tf.get_default_graph())
extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

n_epochs = 40
batch_size = 100

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    X_batch, y_batch = None, None
    for epoch in range(n_epochs):
        for iteration in range(mnist.train.num_examples // batch_size):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        acc_test = accuracy.eval(feed_dict={X: mnist.test.images, y: mnist.test.labels})
        accuracy_val, loss_val, accuracy_summary_str, loss_summary_str = sess.run(
            [accuracy, loss, accuracy_summary, loss_summary],
            feed_dict={X: mnist.validation.images, y: mnist.validation.labels})
        print(epoch, 'Train Accuracy:', acc_train, 'Test Accuracy:', acc_test,
              'Validation Accuracy:', accuracy_val, 'Loss:', loss_val)
        file_writer.add_summary(accuracy_summary_str, epoch)
        file_writer.add_summary(loss_summary_str, epoch)

    tf.train.Saver().save(sess, "./cnn_model/my_mnist_model")
