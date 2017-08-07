import tensorflow as tf
import numpy as np
import matplotlib
from tensorflow.examples.tutorials.mnist import input_data
matplotlib.use('tkagg')
import matplotlib.pyplot as plt


def generator_input(size):
    return np.random.random([size, 10])


def log(x):
    return tf.log(tf.maximum(x, 1e-5))


# dense_layer
# To-do: activation function implementaion
def dense_layer(input, output_dim, activation='linear', scope='None',
                stddev=1.0):
    with tf.variable_scope(scope or 'dense'):
        w = tf.get_variable(
            'w',
            [input.get_shape()[1], output_dim],
            initializer=tf.random_normal_initializer(stddev=stddev)
        )

        b = tf.get_variable(
            'b',
            [output_dim],
            initializer=tf.constant_initializer(0.01)
        )
        if activation == 'relu':
            return tf.nn.relu(tf.matmul(input, w) + b)
        elif activation == 'sigmoid':
            return tf.sigmoid(tf.matmul(input, w) + b)
        else:
            return tf.matmul(input, w) + b


def optimizer(loss, var_list, algo='adam'):
    learning_rate = 0.001
    step = tf.Variable(0, trainable=False)
    if algo == 'adam':
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(
            loss,
            global_step=step,
            var_list=var_list
            )
    elif algo == 'sgd':
        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(
            loss,
            global_step=step,
            var_list=var_list
            )
    return optimizer


def generator(input, hdim=512):
    h0 = dense_layer(input, hdim, activation='relu', scope='g0')
    h1 = dense_layer(h0, hdim, activation='relu', scope='g1')
    h2 = dense_layer(h1, 784, activation='sigmoid', scope='g2')
    return h2


def discriminator(input, h_dim=512):
    h0 = dense_layer(input, h_dim * 2, scope='d0', activation='relu')
    h1 = dense_layer(h0, h_dim * 2, scope='d1', activation='relu')
    h2 = dense_layer(h1, h_dim * 2, scope='d2', activation='relu')
    h3 = dense_layer(h2, 1, scope='d3', activation='linear',)
    return tf.sigmoid(h3), h3


class GAN(object):
    def __init__(self):
        # Generator Network
        with tf.variable_scope('G'):
            self.x = tf.placeholder(tf.float32, [None, 10])
            self.G = generator(self.x)

        # Discriminator Network
        self.z = tf.placeholder(tf.float32, [None, 784])
        # First Discriminator for real images
        with tf.variable_scope('D'):
            self.D1, self.D1_logits = discriminator(self.z)
        # Second Discriminator for generated images
        with tf.variable_scope('D', reuse=True):
            self.D2, self.D2_logits = discriminator(self.G)

        def sigmoid_cross_entropy_with_logits(x, y):
            try:
                return tf.nn.sigmoid_cross_entropy_with_logits(logits=x,
                                                               labels=y)
            except:
                return tf.nn.sigmoid_cross_entropy_with_logits(logits=x,
                                                               targets=y)

        # Next we define the loss functions of the D and G
        self.d_loss_real = tf.reduce_mean(sigmoid_cross_entropy_with_logits(
                                self.D1_logits,
                                tf.ones_like(self.D1)))
        self.d_loss_fake = tf.reduce_mean(sigmoid_cross_entropy_with_logits(
                                self.D2_logits,
                                tf.zeros_like(self.D2)))
        self.d_loss = self.d_loss_real + self.d_loss_fake
        self.g_loss = tf.reduce_mean(sigmoid_cross_entropy_with_logits(
                                self.D2_logits,
                                tf.ones_like(self.D2)))

        # Making lists of trainable variables/parameters
        vars = tf.trainable_variables()
        self.d_param = [var for var in vars if var.name.startswith('D/')]
        self.g_param = [var for var in vars if var.name.startswith('G/')]

        # Now we generate the optimizers for each loss function
        self.d_opt = optimizer(self.d_loss, self.d_param, algo='adam')
        self.g_opt = optimizer(self.g_loss, self.g_param, algo='adam')

    def train(self, data, n_iter=1000, k_iter=2, minibatch_size=128,
              preview=False):
        print('Starting Training')
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            for i in range(n_iter):
                for j in range(k_iter):
                    batch = data.train.next_batch(minibatch_size)
                    d_loss, d_result = sess.run(
                                [self.d_loss, self.d_opt],
                                {self.z: batch[0],
                                    self.x: generator_input(
                                        len(data.train.images))})
                batch = data.train.next_batch(minibatch_size)
                g_loss, g_result = sess.run(
                                [self.g_loss, self.g_opt],
                                {self.z: batch[0],
                                    self.x: generator_input(
                                        len(data.train.images))})
                if i % 1 == 0:
                    print("Iteration {}:\tD Loss={:.6f}\tG Loss={:.6f}"
                          .format(i, d_loss, g_loss))
                    if preview:
                        output = sess.run(self.G,
                                          {self.x:
                                           generator_input(size=1)})
                        output = np.reshape(output, (28, 28))
                        plt.ion()
                        plt.imshow(output)
                        plt.pause(0.001)
                        plt.axis('off')
                        plt.pause(0.001)
                        plt.title('Iteration {}'.format(i))
                        plt.pause(0.001)
            saver.save(sess, 'mnist-gan')
        if preview:
            plt.show()


if __name__ == '__main__':
    model = GAN()
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    model.train(mnist, n_iter=1000, k_iter=2, preview=True, minibatch_size=256)
