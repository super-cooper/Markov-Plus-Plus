from sys import stdout

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from mpp import Utils, TextRNN

# Load input data
data_files = Utils.get_files('Jokes/raw/')
raw = []
for file in data_files:
    with open(file, 'r', encoding='utf-8') as f:
        raw.append(f.read().split(Utils.INSTANCE_DELIMITER))
raw = sum(raw, [])
print('Jokes Loaded.')
padding = max(len(joke) for joke in raw)
jokes_encoded = []
for i, joke in enumerate(raw):
    stdout.write('\rEncoding Joke {}'.format(i))
    joke_e = [np.uint8(ord(joke[j])) if j < len(joke) else 0 for j in range(padding)]
    jokes_encoded.append(joke_e)
print('\nJokes Encoded.')
del raw

print('Splitting into train and test...')
train, test = train_test_split(jokes_encoded)
print('Done.')

n_inputs = 255
n_neurons = 400
steps = padding
n_outputs = 255
n_layers = 3

# Build the model
model = TextRNN(learning_rate=0.001, verbose=True, logging=True, name='GenRNN')

with tf.name_scope(model.get_name()):
    with tf.name_scope('Input'):
        model.add_input_layer(input_type=tf.float32, input_shape=[None, steps, n_inputs],
                              output_type=tf.uint8, output_shape=[None, steps, n_outputs])
    with tf.name_scope('GRU'):
        model.add_recurrent_layers(n_neurons, n_layers, cell_type=tf.nn.rnn_cell.GRUCell)
    with tf.name_scope('Output'):
        model.add_dense_layer(n_outputs, scope_name='Output')
        model.add_batch_norm_layer(scope_name='Output', momentum=0.9)
    with tf.name_scope('Train'):
        model.close(labels=model.y, logits=model.last_added)
        model.define_training_routine()
    with tf.name_scope('Evaluate'):
        model.add_eval_text(tf.summary.text('Generation', tf.py_func(
            lambda x: ''.join(chr(o) for o in x if o > 0),
            model.outputs,
            tf.string)))

n_epochs = 1000
batch_size = 200
n_batches = len(train) // batch_size
train = model.step

with tf.Session() as sess:
    model.init()
    y_batch = None
    X, y = model.X, model.y
    for epoch in range(n_epochs):
        for _ in range(n_batches):
            y_batch = np.random.choice(train, batch_size)
            train(sess, feed_dict={y: y_batch})
        model.checkpoint(epoch, sess, [model.loss], feed_dict={y: test})
