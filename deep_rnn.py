import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

n_steps = 28
n_inputs = 28
n_neurons = 150
n_outputs = 10
n_layers = 3
keep_prob = tf.placeholder_with_default(1.0, shape=[])

learning_rate = 0.001

with tf.variable_scope('rnn', initializer=tf.variance_scaling_initializer()):
    # First dimension of shape is mini-batch size
    X = tf.placeholder(tf.float32, shape=[None, n_steps, n_inputs])
    y = tf.placeholder(tf.int32, [None])

    cells = [tf.nn.rnn_cell.LSTMCell(num_units=n_neurons, use_peepholes=True)
             for layer in range(n_layers)]
    cells_drop = [tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=keep_prob)
                  for cell in cells]
    multi_layer_cell = tf.contrib.rnn.MultiRNNCell(cells_drop)
    outputs, states = tf.nn.dynamic_rnn(multi_layer_cell, X, dtype=tf.float32)
    top_layer_h_state = states[-1][1]
    logits = tf.layers.dense(top_layer_h_state, n_outputs, name="softmax")

    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y))
    train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

# Load MNIST data
mnist = input_data.read_data_sets('/tmp/data')
X_test = mnist.test.images.reshape((-1, n_steps, n_inputs))
y_test = mnist.test.labels

batch_size = 150
n_epochs = 10
keep_train_prob = 0.5
saver = tf.train.Saver()

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    X_batch, y_batch = None, None
    for epoch in range(n_epochs):
        for iteration in range(mnist.train.num_examples // batch_size):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            X_batch = X_batch.reshape((batch_size, n_steps, n_inputs))
            sess.run(train, feed_dict={X: X_batch, y: y_batch, keep_prob: keep_train_prob})
        acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        acc_test = accuracy.eval(feed_dict={X: X_test, y: y_test})
        print(epoch, 'Train accuracy:', acc_train, 'Test accuracy:', acc_test)

    saver.save(sess, './deep_model/model')
