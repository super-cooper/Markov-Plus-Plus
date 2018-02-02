import os

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def xavier_init(size):
    input_dim = size[0]
    xavier_variance = 1. / tf.sqrt(input_dim/2.)
    return tf.random_normal(shape=size, stddev=xavier_variance)


def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig


# Random noise setting for Generator
# Takes a 100-dimensional vector from random distribution and returns a 768-dimensional vector (MNIST image)
Z = tf.placeholder(tf.float32, shape=[None, 100], name='Z')

# Generator parameter settings
G_W1 = tf.Variable(xavier_init([100, 128]), name='G_W1')
G_b1 = tf.Variable(tf.zeros(shape=[128]), name='G_b1')
G_W2 = tf.Variable(xavier_init([128, 784]), name='G_W2')
G_b2 = tf.Variable(tf.zeros(shape=[784]), name='G_b2')
theta_G = [G_W1, G_W2, G_b1, G_b2]

# Input Image MNIST setting for Discriminator [28x28=784]
X = tf.placeholder(tf.float32, shape=[None, 784], name='X')

# Discriminator parameter settings
D_W1 = tf.Variable(xavier_init([784, 128]), name='D_W1')
D_b1 = tf.Variable(tf.zeros(shape=[128]), name='D_b1')
D_W2 = tf.Variable(xavier_init([128, 1]), name='D_W2')
D_b2 = tf.Variable(tf.zeros(shape=[1]), name='D_b2')

theta_D = [D_W1, D_W2, D_b1, D_b2]


# Generator Network
def generator(z):
    G_h1 = tf.nn.relu(tf.matmul(z, G_W1) + G_b1)
    G_log_prob = tf.matmul(G_h1, G_W2) + G_b2
    G_prob = tf.nn.sigmoid(G_log_prob)

    return G_prob


# Discriminator Network
def discriminator(x, z=None):
    D_h1 = tf.nn.relu(tf.matmul(x, D_W1) + D_b1)
    D_logit = tf.matmul(D_h1, D_W2) + D_b2

    return D_logit


tfgan = tf.contrib.gan

model = tfgan.gan_model(generator, discriminator, real_data=X, generator_inputs=Z)

G_sample = model.generated_data

# Loss functions from the paper
loss = tfgan.gan_loss(model,
                      generator_loss_fn=tfgan.losses.wasserstein_generator_loss,
                      discriminator_loss_fn=tfgan.losses.wasserstein_discriminator_loss)

D_loss, G_loss = -loss.discriminator_loss, -loss.generator_loss


# Update D(X)'s parameters
D_solver = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5).minimize(D_loss, var_list=theta_D)

# Update G(Z)'s parameters
G_solver = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5).minimize(G_loss, var_list=theta_G)


def sample_Z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])


batch_size = 128
Z_dim = 100

sess = tf.Session()
sess.run(tf.global_variables_initializer())

mnist = input_data.read_data_sets('tmp/MNIST/', one_hot=True)

if not os.path.exists('./simple_gan_output/'):
    os.makedirs('./simple_gan_output/')

i = 0

for itr in range(1000000):
    if itr % 1000 == 0:
        samples = sess.run(G_sample, feed_dict={Z: sample_Z(16, Z_dim)})

        fig = plot(samples)
        plt.savefig('./simple_gan_output/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
        i += 1
        plt.close(fig)

    X_mb, _ = mnist.train.next_batch(batch_size)

    _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={X: X_mb, Z: sample_Z(batch_size, Z_dim)})
    _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={Z: sample_Z(batch_size, Z_dim)})

    if itr % 1000 == 0:
        print('Iter: {}'.format(itr))
        print('D loss: {:.4}'. format(D_loss_curr))
        print('G_loss: {:.4}'.format(G_loss_curr))
        print()
