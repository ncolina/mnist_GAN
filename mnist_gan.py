import tensorflow as tf
import numpy as np
import matplotlib
import time
from tensorflow.examples.tutorials.mnist import input_data
matplotlib.use('tkagg')
import matplotlib.pyplot as plt


def generator_input(size):
    return np.random.random([size, 100])


def log(x):
    return tf.log(tf.maximum(x, 1e-5))


def leaky_relu(x):
    return tf.maximum(0.01*x, x)


def minibatch_disc_layer(input, num_kernels=5, kernel_dim=3, scope=None):
    with tf.variable_scope(scope or 'minibatch_disc'):
        x = dense_layer(input, num_kernels * kernel_dim)
        activation = tf.reshape(x, (-1, num_kernels, kernel_dim))
        diffs = tf.expand_dims(activation, 3) - \
            tf.expand_dims(tf.transpose(activation, [1, 2, 0]), 0)
        abs_diffs = tf.reduce_sum(tf.abs(diffs), 2)
        minibatch_features = tf.reduce_sum(tf.exp(-abs_diffs), 2)
        return tf.concat([input, minibatch_features], 1)


# dense_layer
# To-do: activation function implementaion
def dense_layer(input, output_dim, activation=None, scope=None,
                stddev=0.02, batch_norm=False, training=True):
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
        output = tf.matmul(input, w) + b
        if batch_norm:
            output = tf.contrib.layers.batch_norm(output, scale=True,
                                                  is_training=training,
                                                  updates_collections=None)

        if activation == 'relu':
            return tf.nn.relu(output)
        elif activation == 'sigmoid':
            return tf.sigmoid(output)
        elif activation == 'leakyrelu':
            return leaky_relu(output)
        else:
            return tf.matmul(input, w) + b


def conv_layer(input, filter, stride, output_shape=None, inverse=False,
               activation=None, batch_norm=False, scope=None, training=True):

        def conv_2d(input, w, inverse=False):
            if inverse:
                return tf.nn.conv2d_transpose(input, w,
                                              output_shape=output_shape,
                                              strides=stride)
            else:
                return tf.nn.conv2d(input, w, strides=stride,
                                    padding='SAME')
        output_dim = filter[-2] if inverse else filter[-1]
        with tf.variable_scope(scope or 'conv'):
            w = tf.get_variable(
                'w',
                filter,
                initializer=tf.random_normal_initializer(stddev=0.02)
            )
            b = tf.get_variable(
                'b',
                [output_dim],
                initializer=tf.constant_initializer(0.01)
            )
            conv = conv_2d(input, w, inverse=inverse) + b

            if batch_norm:
                conv = tf.contrib.layers.batch_norm(conv, scale=True,
                                                    is_training=training,
                                                    updates_collections=None)
            if activation == 'leakyrelu':
                return leaky_relu(conv + b)
            elif activation == 'relu':
                return tf.nn.relu(conv + b)
            else:
                return conv + b


def optimizer(loss, var_list, algo='adam'):
    learning_rate = 0.0002
    beta = 0.5
    step = tf.Variable(0, trainable=False)
    if algo == 'adam':
        optimizer = tf.train.AdamOptimizer(learning_rate, beta1=beta).minimize(
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


def generator(input, training=True, hdim=1024, minibatch_disc=False):
    h0 = dense_layer(input, hdim, activation='leakyrelu',
                     scope='dense0', training=training)
    if minibatch_disc:
        h0 = minibatch_disc_layer(h0)
    h1 = dense_layer(h0, 7*7*64, activation='leakyrelu',
                     scope='dense1', training=training)
    h1_reshape = tf.reshape(h1, [-1, 7, 7, 64])
    batch_size = tf.shape(h1_reshape)[0]
    deconv_shape = [batch_size, 14, 14, 32]
    h2 = conv_layer(h1_reshape, filter=[5, 5, 32, 64], stride=[1, 2, 2, 1],
                    output_shape=deconv_shape,
                    inverse=True, activation='leakyrelu',
                    scope='conv2', training=training,
                    batch_norm=True)
    batch_size = tf.shape(h2)[0]
    deconv_shape = [batch_size, 28, 28, 1]
    h3 = conv_layer(h2, filter=[5, 5, 1, 32], stride=[1, 2, 2, 1],
                    output_shape=deconv_shape,
                    inverse=True, activation=None,
                    scope='conv3', training=training,
                    batch_norm=False)
    h3 = tf.reshape(h3, [-1, 784])
    return tf.sigmoid(h3), h3


def discriminator(input, training=True, h_dim=1024, keep_prob=0.5,
                  minibatch_disc=False):
    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='SAME')
    input = tf.reshape(input, [-1, 28, 28, 1])
    h0 = conv_layer(input, [5, 5, 1, 128], [1, 1, 1, 1], activation='leakyrelu',
                    scope='conv0', training=training, batch_norm=True)
    hpool_0 = max_pool_2x2(h0)

    h1 = conv_layer(hpool_0, [5, 5, 128, 64], [1, 1, 1, 1],
                    activation='leakyrelu', scope='conv1',
                    training=training, batch_norm=True)
    hpool_1 = max_pool_2x2(h1)
    hflat_1 = tf.reshape(hpool_1, [-1, 7*7*64])
    if minibatch_disc:
        hflat_1 = minibatch_disc_layer(hflat_1)
    h2 = dense_layer(hflat_1, h_dim, scope='dense2', activation='leakyrelu',
                     batch_norm=True, training=training)
    h2_drop = tf.nn.dropout(h2, keep_prob)
    h3 = dense_layer(h2_drop, h_dim, scope='dense3', activation='leakyrelu',
                     batch_norm=True, training=training)
    h3_drop = tf.nn.dropout(h3, keep_prob)
    h4 = dense_layer(h3_drop, 1, scope='dense4', activation=None,
                     batch_norm=False, training=training)
    return tf.sigmoid(h4), h4


class GAN(object):
    def __init__(self):
        self.training = tf.placeholder(tf.bool, name='training')

        # Generator Network
        with tf.variable_scope('G'):
            self.x = tf.placeholder(tf.float32, [None, 100], name='g-input')
            self.G = generator(self.x, training=self.training,
                               minibatch_disc=True)

        # Discriminator Network
        self.z = tf.placeholder(tf.float32, [None, 784], name='d-input')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        # First Discriminator for real images
        with tf.variable_scope('D'):
            self.D1, self.D1_logits = discriminator(self.z,
                                                    training=self.training,
                                                    keep_prob=self.keep_prob,
                                                    minibatch_disc=True)
        # Second Discriminator for generated images
        with tf.variable_scope('D', reuse=True):
            self.D2, self.D2_logits = discriminator(self.G,
                                                    training=self.training,
                                                    keep_prob=self.keep_prob,
                                                    minibatch_disc=True)

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
        if preview:  # initialize plotting environment
            fig, ax = plt.subplots(2, 2)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            g_hist = []
            d_hist = []
            tf.get_default_graph().finalize()

            for i in range(n_iter):
                start = time.time()
                for j in range(k_iter):
                    batch = data.train.next_batch(minibatch_size)
                    d_loss, d_result = sess.run(
                                            [self.d_loss, self.d_opt],
                                            {self.z: batch[0],
                                             self.x: generator_input(
                                                    minibatch_size),
                                             self.training: True,
                                             self.keep_prob: 0.5})
                    d_hist.append(d_loss)

                batch = data.train.next_batch(minibatch_size)
                g_loss, g_result = sess.run(
                                [self.g_loss, self.g_opt],
                                {self.z: batch[0],
                                    self.x: generator_input(
                                        minibatch_size),
                                    self.training: True,
                                    self.keep_prob: 1.0})
                g_hist.append(g_loss)
                iter_time = time.time() - start
                if i % 1 == 0:
                    print("Iteration {}:\tTime={:.2f}s\tD Loss={:.6f}\tG Loss={:.6f}"
                          .format(i, iter_time, d_loss, g_loss))
                    if preview:
                        output = sess.run(self.G,
                                          {self.x:
                                           generator_input(size=1),
                                           self.training: False,
                                           self.keep_prob: 1.0})
                        output = np.reshape(output, (-1, 28, 28))
                        plt.ion()
                        ax[0][0].imshow(output[0], cmap='Greys')
                        ax[0][1].plot(g_hist)
                        ax[1][1].plot(d_hist)
                        plt.pause(0.001)
                        ax[0][0].axis('off')
                        ax[1][0].axis('off')
                        plt.pause(0.001)
                        ax[0][0].set_title('Iteration {}'.format(i))
                        ax[0][1].set_title('G Loss')
                        ax[1][1].set_title('D Loss')
                        plt.pause(0.001)
                saver.save(sess, 'mnist-gan')
        if preview:
            plt.show()
        return (np.array(d_hist), np.array(g_hist))


if __name__ == '__main__':
    model = GAN()
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    hist = model.train(mnist, n_iter=20000, k_iter=2,
                       preview=True, minibatch_size=128)
