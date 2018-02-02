import os
from datetime import datetime
from functools import partial

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def log_dir(prefix=""):
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    root_log_dir = "dnn_model/tb"
    if prefix:
        prefix += "-"
    name = prefix + "run-" + now
    return "{}/{}/".format(root_log_dir, name)


n_inputs = 28 * 28  # MNIST image dimensions
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 10

# X_ (training input) is a 2D tensor containing the 28*28 matrix of an MNIST image.
# However, we don't know how many instances each training batch will contain
# Therefore, the shape of X_ is (?, n_inputs)
X = tf.placeholder(tf.float32, shape=(None, n_inputs), name='X_')

# We know that y (targets) is a 1D tensor with one entry per instance, but we don't know the size of the training batch
# Therefore, the shape of y is (?)
y = tf.placeholder(tf.int64, shape=None, name='y')

# This variable will tell batch norm if we're training or not
training = tf.placeholder_with_default(False, shape=(), name='training')
# We're using He initialization for our weights
he_init = tf.variance_scaling_initializer()

# Now it's time to create the DNN
with tf.name_scope('DNN'):
    bn_layer = partial(tf.layers.batch_normalization, training=training, momentum=0.9)
    dense = partial(tf.layers.dense, kernel_initializer=he_init)
    hidden_1 = dense(X, n_hidden1, name='Hidden1')
    act_1 = tf.nn.selu(bn_layer(hidden_1))
    hidden_2 = dense(act_1, n_hidden2, name='Hidden2')
    act_2 = tf.nn.selu(bn_layer(hidden_2))
    logits = bn_layer(dense(act_2, n_outputs, name='Output'))

# We're going to use an error function called sparse_softmax_cross_entropy_with_logits
# It will compute cross-entropy on the output of the network before going through the softmax function
# It expects labels in the form of integers ranging from 0 to n_classes - 1
# This gives us an ID tensor containing the cross-entropy for each instance
# We can then use TensorFlow's reduce_mean function to compute the mean cross-entropy over all instances
# Also note that softmax_cross_entropy_with_logits takes one-hot vectors instead of integers
with tf.name_scope('loss'):
    x_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(x_entropy, name='loss')
    loss_summary = tf.summary.scalar('log_loss', loss)

# Now we have our model's architecture set. We need to define a backpropagation function
learning_rate = 0.001
with tf.name_scope('train'):
    optimizer = tf.train.AdamOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)

# Now we will build an op to evaluate our network by checking to see if the target has the highest probability
# in the output tensor
with tf.name_scope('eval'):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    accuracy_summary = tf.summary.scalar('accuracy', accuracy)

# Now we'll set up some utilities for working with the model
saver = tf.train.Saver()

# Now it's time to load the data for the execution phase
mnist = input_data.read_data_sets('/tmp/data')

# Let's define the parameters for running the network
m, n = mnist.train.images.shape
n_epochs = 10_000
batch_size = 200
n_batches = int(np.ceil(m / batch_size))

# Some logging paths
checkpoint_path = "/tmp/my_deep_mnist_model.ckpt"
checkpoint_epoch_path = checkpoint_path + ".epoch"
final_model_path = "./my_deep_mnist_model"
file_writer = tf.summary.FileWriter(log_dir('mnist_dnn'), tf.get_default_graph())

# Starting values for early stopping
best_loss = np.inf
epochs_without_progress = 0
max_epochs_without_progress = 50

# Explicit ops for batch normalization
extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

# Let's get our test and validation set
X_test, y_test = mnist.test.images, mnist.test.labels
X_valid, y_valid = mnist.validation.images, mnist.validation.labels

# Now let's run it!
with tf.Session() as sess:
    if os.path.isfile(checkpoint_epoch_path):
        # if the checkpoint file exists, restore the model and load the epoch number
        with open(checkpoint_epoch_path, "rb") as f:
            start_epoch = int(f.read())
        print("Training was interrupted. Continuing at epoch", start_epoch)
        saver.restore(sess, checkpoint_path)
    else:
        start_epoch = 0
        tf.global_variables_initializer().run()

    X_batch, y_batch = None, None
    for epoch in range(n_epochs):
        # Iterate through a number of mini-batches (enough to cover the whole training set)
        for iteration in range(n_batches):
            # Get our next batch of training data and targets
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            # Run the network on the batch
            sess.run([training_op, extra_update_ops], feed_dict={X: X_batch, y: y_batch, training: True})
        accuracy_val, loss_val, accuracy_summary_str, loss_summary_str = sess.run(
            [accuracy, loss, accuracy_summary, loss_summary],
            feed_dict={X: X_valid, y: y_valid})
        # Evaluate accuracy on both the training set and the test set
        # acc_train = accuracy.eval(feed_dict={X_: X_batch, y: y_batch})
        # acc_test = accuracy.eval(feed_dict={X_: X_test, y: y_test})
        file_writer.add_summary(accuracy_summary_str, epoch)
        file_writer.add_summary(loss_summary_str, epoch)

        # print(epoch, 'Train accuracy:', acc_train, 'Test accuracy:', acc_test)
        if epoch % 5 == 0:
            print("Epoch:", epoch,
                  "\tValidation accuracy: {:.3f}%".format(accuracy_val * 100),
                  "\tLoss: {:.5f}".format(loss_val))
            saver.save(sess, checkpoint_path)
            with open(checkpoint_epoch_path, "wb") as f:
                f.write(b"%d" % (epoch + 1))
            if loss_val < best_loss:
                saver.save(sess, final_model_path)
                best_loss = loss_val
            else:
                epochs_without_progress += 5
                if epochs_without_progress > max_epochs_without_progress:
                    print("Early stopping")
                    break

    # save_path = saver.save(sess, './dnn_model/my_model_final.cpkt')
os.remove(checkpoint_epoch_path)
with tf.Session() as sess:
    saver.restore(sess, final_model_path)
    accuracy_val = accuracy.eval(feed_dict={X: X_test, y: y_test, training: False})
print('Best model accuracy:', accuracy_val)
file_writer.close()
