import tensorflow as tf
from mpp import TextNet
from tensorflow.examples.tutorials.mnist import input_data

# Some hyperparameters
n_inputs = 28 * 28  # MNIST image dimensions
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 10

# Define the model architecture
model = TextNet(learning_rate=0.001, logging=True, verbose=True, name='SimpleDNN')
with tf.name_scope(model.get_name()):
    with tf.name_scope('Input'):
        model.add_input_layer(input_type=tf.float32, input_shape=[None, n_inputs],
                              output_type=tf.int64, output_shape=[None])
    with tf.name_scope('Hidden1'):
        model.add_dense_layer(n_hidden1, scope_name='Hidden1')
        model.add_batch_norm_layer(scope_name='Hidden1', momentum=0.9)
        model.activate(scope_name='Hidden1')
    with tf.name_scope('Hidden2'):
        model.add_dense_layer(n_hidden2, scope_name='Hidden2')
        model.add_batch_norm_layer(scope_name='Hidden2', momentum=0.9)
        model.activate(scope_name='Hidden2')
    with tf.name_scope('Output'):
        model.add_dense_layer(n_outputs, scope_name='Output')
        model.add_batch_norm_layer(scope_name='Output', momentum=0.9)
    with tf.name_scope('Train'):
        model.close(labels=model.y, logits=model.last_added)
        model.define_training_routine()
    with tf.name_scope('Eval'):
        model.add_eval_accuracy()

# Now it's time to load the data for the execution phase
mnist = input_data.read_data_sets('/tmp/data')

# Let's define the parameters for running the network
m, n = mnist.train.images.shape
n_epochs = 50
batch_size = 200
n_batches = m // batch_size
train = model.step

# Let's get our test and validation set
X_test, y_test = mnist.test.images, mnist.test.labels
X_valid, y_valid = mnist.validation.images, mnist.validation.labels

# Pray to jeebus this works
with tf.Session() as sess:
    model.init()
    x_batch, y_batch = None, None
    X, y = model.X, model.y
    for epoch in range(n_epochs):
        for _ in range(n_batches):
            # Get our next batch of training data and targets
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            # Run the network on the batch
            train(sess, feed_dict={X: X_batch, y: y_batch})
        model.checkpoint(epoch, sess, [model.accuracy, model.loss], feed_dict={X: X_valid, y: y_valid})
model.logger.flush()
