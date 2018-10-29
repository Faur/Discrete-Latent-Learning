import tensorflow as tf
import numpy as np
import glob, random, os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

_EXP_NAME = "continuous"
model_path = "saved_models/"
model_name = model_path + _EXP_NAME + '_model'

_EMBEDDING_SIZE = 32  # TODO: Handle this better!

class Network(object):
    # Create model
    def __init__(self, input_dim=(28, 28, 1), N=32, M=2, lr=0.00001):
        self.num_img_sum = 3

        self.img_channels = input_dim[-1]
        self.image = tf.placeholder(tf.float32, (None,)+input_dim, name='image')
        tf.summary.image('image', self.image, self.num_img_sum)

        self.latent_logits = self.encoder(self.image)
        self.z_mu, self.z_logvar = self.latent(self.latent_logits)
        self.z = self.sample_z(self.z_mu, self.z_logvar)
        self.reconstructions = self.decoder(self.z)
        tf.summary.image('reconstructions', self.reconstructions, self.num_img_sum)

        self.loss = self.compute_loss()
        self.merged = tf.summary.merge_all()

    def sample_z(self, mu, logvar):
        eps = tf.random_normal(shape=tf.shape(mu))
        return mu + tf.exp(logvar / 2) * eps

    def encoder(self, x):
        print('Encoder')
        print(x)
        x = tf.layers.conv2d(x, filters=32, kernel_size=4, strides=2, padding='valid', activation=tf.nn.relu)
        print(x)
        x = tf.layers.conv2d(x, filters=64, kernel_size=4, strides=2, padding='valid', activation=tf.nn.relu)
        print(x)
        x = tf.layers.conv2d(x, filters=128, kernel_size=4, strides=2, padding='valid', activation=tf.nn.relu)
        print(x)
        # x = tf.layers.conv2d(x, filters=256, kernel_size=4, strides=2, padding='valid', activation=tf.nn.relu)
        print()
        return x

    def latent(self, x):
        print("Latent: Continuous")
        x = tf.layers.flatten(x)
        print(x)
        z_mu = tf.layers.dense(x, units=_EMBEDDING_SIZE, name='z_mu')
        z_logvar = tf.layers.dense(x, units=_EMBEDDING_SIZE, name='z_logvar')
        print(z_mu)
        print(z_logvar)
        print()
        return z_mu, z_logvar

    def decoder(self, z):
        print('Decoder')
        x = tf.layers.dense(z, 1024, activation=None)
        print(x)
        x = tf.reshape(x, [-1, 1, 1, 1024])
        print(x)
        x = tf.layers.conv2d_transpose(x, filters=128, kernel_size=5, strides=2, padding='valid', activation=tf.nn.relu)
        print(x)
        x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=5, strides=2, padding='valid', activation=tf.nn.relu)
        print(x)
        x = tf.layers.conv2d_transpose(x, filters=self.img_channels, kernel_size=4, strides=2, padding='valid', activation=tf.nn.relu)
        print(x)
        print()
        return x

    def compute_loss(self):
        logits_flat = tf.layers.flatten(self.reconstructions)
        labels_flat = tf.layers.flatten(self.image)
        reconstruction_loss = tf.reduce_sum(tf.square(logits_flat - labels_flat), axis = 1)
        kl_loss = 0.5 * tf.reduce_sum(tf.exp(self.z_logvar) + self.z_mu**2 - 1. - self.z_logvar, 1)
        vae_loss = tf.reduce_mean(reconstruction_loss + kl_loss)
        tf.summary.scalar("KL_loss", tf.reduce_mean(kl_loss))
        tf.summary.scalar("rec_loss", tf.reduce_mean(reconstruction_loss))
        tf.summary.scalar("total_loss", vae_loss)
        return vae_loss

    def get_embedding(self, sess, observation):
        return sess.run(self.z, feed_dict={self.image: observation[None, :, :, :]})

    def normalize_observation(self, observation):
        return observation.astype('float32') / 255.

    def predict(self, sess, data):
        print(sess)
        pred, mu, z_logvar, z = sess.run([self.reconstructions, self.z_mu, self.z_logvar, self.z], feed_dict={self.image:data})
        sigma = np.exp(z_logvar/2)
        return pred, mu, sigma, z


def data_iterator(data, batch_size):
    N = data.shape[0]
    epoch = 0
    while True:
        np.random.shuffle(data)
        for i in range(int(N/batch_size)):
            yield epoch, i, data[i*batch_size:(i+1)*batch_size]
        epoch + 1

def train_vae():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.InteractiveSession(config=config)

    # TODO: load or inferr gloabl step.
    global_step = tf.Variable(0, name='global_step', trainable=False)

    writer = tf.summary.FileWriter('logdir/'+_EXP_NAME)

    network = Network()
    train_op = tf.train.AdamOptimizer(0.001).minimize(network.loss, global_step=global_step)
    tf.global_variables_initializer().run()

    saver = tf.train.Saver(max_to_keep=1)
    step = global_step.eval()

    ### DATA
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = np.expand_dims(x_train/255., -1)
    x_test = np.expand_dims(x_test/255., -1)

    batch_size = 32
    batch_per_epoch = int(x_train.shape[0]/batch_size)
    train_iter = data_iterator(x_train, batch_size=batch_size)
    test_iter = data_iterator(x_test, batch_size=batch_size)

    try:
        saver.restore(sess, tf.train.latest_checkpoint(model_path))
        print("Model restored from: {}".format(model_path))
    except:
        print("Could not restore saved model")

    print("\nBegin Training")
    try:
        while True:
            epoch, e_step, images = next(train_iter)

            _, loss_value = sess.run([train_op, network.loss],
                                feed_dict={network.image: images})

            if np.isnan(loss_value):
                raise ValueError('Loss value is NaN')
            if step % 10 == 0 and step > 0:
                print("E {:5}, {:5}/{}: training loss {:.6f}".format(
                    epoch, e_step, batch_per_epoch, loss_value))
                [summary] = sess.run([network.merged], feed_dict={network.image: images})
                writer.add_summary(summary, step)
                save_path = saver.save(sess, model_name, global_step=global_step)

            step += 1

    except (KeyboardInterrupt, SystemExit):
        print("Manual Interrupt")

    except Exception as e:
        print("Exception: {}".format(e))


def load_vae():
    graph = tf.Graph()
    with graph.as_default():
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config, graph=graph)

        network = Network()
        init = tf.global_variables_initializer()
        sess.run(init)

        saver = tf.train.Saver(max_to_keep=1)
        training_data = data_iterator(batch_size=128)

        try:
            saver.restore(sess, tf.train.latest_checkpoint(model_path))
        except:
            raise ImportError("Could not restore saved model")

        return sess, network

if __name__ == '__main__':
    train_vae()