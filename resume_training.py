import mnist_gan

def restore_session(path):
    saver = tf.train.Saver()
    sess = tf.Session()
    saver.restore(sess, tf.train.latest_checkpoint(path))
    return sess

if __name__ == '__main__':
    model = mnist_gan.GAN()
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    hist = model.train(mnist, n_iter=20000, k_iter=1,
                       preview=True, minibatch_size=32, resume=True)
