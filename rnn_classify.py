import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

from mpp import TextRNN

data = pd.read_csv('Spam/spam.csv')

train, test = train_test_split(data)
print('Data loaded')
print('Encoding data')
train_data = np.array([[ord(c) for c in s] for s in train['v2']])
train_labels = np.array([1 if 'ham' in c else 0 for c in train['v1']])
test_data = np.array([[ord(c) for c in s] for s in test['v2']])
test_labels = np.array([1 if 'ham' in c else 0 for c in test['v1']])

padding = max(len(s) for s in np.concatenate((train_data, test_data), axis=0))
for i in range(len(train_data)):
    while len(train_data[i]) < padding:
        train_data[i].append(0)
for i in range(len(train_data)):
    while len(train_data[i]) < padding:
        train_data[i].append(0)

print('Done.')

n_steps = padding
n_inputs = 1
n_neurons = 400
n_outputs = 1
n_layers = 3

model = TextRNN(learning_rate=0.001, verbose=True, logging=True, name='RNNClassifier')

with tf.name_scope('RNNClassifier'):
    with tf.name_scope('Input'):
        model.add_input_layer(input_type=tf.int32, input_shape=[None, n_steps, n_inputs],
                              output_type=tf.float32, output_shape=[None])
    with tf.name_scope('GRU'):
        model.add_recurrent_layers(n_neurons, n_layers, use_peepholes=True)
    with tf.name_scope('Output'):
        model.add_dense_layer(n_outputs, scope_name='Output')
        model.add_batch_norm_layer(scope_name='Output', momentum=0.9)
    with tf.name_scope('Train'):
        model.close(logits=model.last_added, labels=model.y)
        model.define_training_routine()
    with tf.name_scope('Evaluate'):
        model.add_eval_accuracy()

n_epochs = 1000
batch_size = 150
n_batches = len(train_data) // batch_size
train = model.step

with tf.Session() as sess:
    model.init()
    x_batch, y_batch = None, None
    X, y = model.X, model.y
    for epoch in range(n_epochs):
        for _ in range(n_batches):
            indices = np.random.choice(len(train_data), batch_size)
            X_batch, y_batch = train_data[indices], train_labels[indices]
            train(sess, feed_dict={X: X_batch, y: y_batch})
        model.checkpoint(epoch, sess, [model.accuracy, model.loss], feed_dict={X: test_data, y: test_labels})
model.logger.flush()
