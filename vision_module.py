import tensorflow as tf
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# np.random.seed(0)
# tf.set_random_seed(0)

class BaseAutoEncoder(object):
    # Create model
    def __init__(self, embedding_dim, input_dim=(28, 28, 1), lr=0.00001):
        self.embedding_dim = embedding_dim
        self.tb_num_images = 3

        self.img_channels = input_dim[-1]
        self.image = tf.placeholder(tf.float32, (None,)+input_dim, name='image')
        tf.summary.image('image', self.image, self.tb_num_images)

        self.latent_logits = self.encoder(self.image)
        self.z, self.latent_var = self.latent(self.latent_logits)
        self.reconstructions = self.decoder(self.z)
        tf.summary.image('reconstructions', self.reconstructions, self.tb_num_images)

        self.loss = self.compute_loss()
        self.merged = tf.summary.merge_all()

    def encoder(self, x):
        print('Encoder')
        print(x)
        x = tf.layers.conv2d(x, filters=32, kernel_size=4, strides=2, padding='valid', activation=tf.nn.relu)
        print(x)
        x = tf.layers.conv2d(x, filters=64, kernel_size=4, strides=2, padding='valid', activation=tf.nn.relu)
        print(x)
        x = tf.layers.conv2d(x, filters=128, kernel_size=4, strides=2, padding='valid', activation=tf.nn.relu)
        print(x)
        print()
        return x

    def decoder(self, z):
        print('Decoder')
        decoder_input_size = 128
        x = tf.layers.dense(z, decoder_input_size, activation=None)
        print(x)
        x = tf.reshape(x, [-1, 1, 1, decoder_input_size])
        print(x)
        x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=5, strides=2, padding='valid', activation=tf.nn.relu)
        print(x)
        x = tf.layers.conv2d_transpose(x, filters=32, kernel_size=5, strides=2, padding='valid', activation=tf.nn.relu)
        print(x)
        x = tf.layers.conv2d_transpose(x, filters=16, kernel_size=4, strides=2, padding='valid', activation=tf.nn.relu)
        print(x)
        x = tf.layers.conv2d(x, filters=self.img_channels, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu)
        print(x)
        print()
        return x

    def sample_z(self, *args):
        raise NotImplementedError

    def latent(self, x):
        raise NotImplementedError

    def get_embedding(self, sess, observation):
        raise NotImplementedError

    def predict(self, sess, data):
        raise NotImplementedError

    def compute_loss(self):
        raise NotImplementedError


class ContinuousAutoEncoder(BaseAutoEncoder):
    def sample_z(self, mu, logvar):
        eps = tf.random_normal(shape=tf.shape(mu))
        return mu + tf.exp(logvar / 2) * eps

    def latent(self, x):
        print("Latent: Continuous")

        x = tf.layers.flatten(x)
        print(x)
        z_mu = tf.layers.dense(x, units=self.embedding_dim[0], name='z_mu')
        z_logvar = tf.layers.dense(x, units=self.embedding_dim[0], name='z_logvar')
        print(z_mu)
        print(z_logvar)
        z = self.sample_z(z_mu, z_logvar)
        print(z)
        print()
        return z, (z_mu, z_logvar)

    def compute_loss(self):
        z_mu, z_logvar = self.latent_var

        logits_flat = tf.layers.flatten(self.reconstructions)
        labels_flat = tf.layers.flatten(self.image)

        reconstruction_loss = tf.reduce_sum(tf.square(logits_flat - labels_flat), axis = 1)
        kl_loss = 0.5 * tf.reduce_sum(tf.exp(z_logvar) + z_mu**2 - 1. - z_logvar, 1)
        vae_loss = tf.reduce_mean(reconstruction_loss + kl_loss)
        tf.summary.scalar("train/KL_loss_c", tf.reduce_mean(kl_loss))
        tf.summary.scalar("train/rec_loss", tf.reduce_mean(reconstruction_loss))
        tf.summary.scalar("train/total_loss_c", vae_loss)
        return vae_loss

    def predict(self, sess, data):
        print(sess)
        z_mu, z_logvar = self.latent_var
        pred, mu, z_logvar, z = sess.run([self.reconstructions, z_mu, z_logvar, self.z], feed_dict={self.image:data})
        sigma = np.exp(z_logvar/2)
        return pred, mu, sigma, z

    def get_embedding(self, sess, observation):
        return sess.run(self.z, feed_dict={self.image: observation[None, :, :, :]})


