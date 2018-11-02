import tensorflow as tf
import numpy as np
import os

from keras import backend as K
from keras.activations import softmax

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class BaseAutoEncoder(object):
    # Create model
    def __init__(self, latent_dim, input_dim=(28, 28, 1), lr=0.00001):
        self.latent_dim = latent_dim
        self.tb_num_images = 3

        self.img_channels = input_dim[-1]
        self.image = tf.placeholder(tf.float32, (None,)+input_dim, name='image')
        tf.summary.image('image', self.image, self.tb_num_images)

    def setup_network(self):
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

    def reconstruction_loss(self):
        logits_flat = tf.layers.flatten(self.reconstructions)
        labels_flat = tf.layers.flatten(self.image)
        return tf.reduce_sum(tf.square(logits_flat - labels_flat), axis=1)

    def print_summary(self):
        print()

    def update_params(self, *args, **kwargs):
        pass

    def sample_z(self, *args):
        raise NotImplementedError

    def latent(self, x):
        raise NotImplementedError

    def compute_loss(self):
        raise NotImplementedError

    def get_embedding(self, sess, observation):
        raise NotImplementedError

    def predict(self, sess, data):
        raise NotImplementedError


class ContinuousAutoEncoder(BaseAutoEncoder):
    def __init__(self, *args, **kwargs):
        super(ContinuousAutoEncoder, self).__init__(*args, **kwargs)
        self.setup_network()

    def sample_z(self, mu, logvar):
        eps = tf.random_normal(shape=tf.shape(mu))
        return mu + tf.exp(logvar / 2) * eps

    def latent(self, x):
        print("Latent: Continuous")

        x = tf.layers.flatten(x)
        print(x)
        z_mu = tf.layers.dense(x, units=self.latent_dim[0], name='z_mu')
        z_logvar = tf.layers.dense(x, units=self.latent_dim[0], name='z_logvar')
        print(z_mu)
        print(z_logvar)
        z = self.sample_z(z_mu, z_logvar)
        print(z)
        print()

        std = tf.sqrt(tf.exp(z_logvar))
        tf.summary.histogram('train_C/z_mu', z_mu)
        tf.summary.histogram('train_C/z_std', std)
        tf.summary.histogram('train_C/z', z)

        return z, (z_mu, z_logvar)

    def compute_loss(self):
        z_mu, z_logvar = self.latent_var

        rec_loss = self.reconstruction_loss()
        kl_loss = 0.5 * tf.reduce_sum(tf.exp(z_logvar) + z_mu**2 - 1. - z_logvar, 1)
        vae_loss = tf.reduce_mean(rec_loss + kl_loss)
        tf.summary.scalar("train/KL_loss", tf.reduce_mean(kl_loss))
        tf.summary.scalar("train/rec_loss", tf.reduce_mean(rec_loss))
        tf.summary.scalar("train/total_loss", vae_loss)
        return vae_loss

    def predict(self, sess, data):
        print(sess)
        z_mu, z_logvar = self.latent_var
        pred, mu, z_logvar, z = sess.run([self.reconstructions, z_mu, z_logvar, self.z], feed_dict={self.image:data})
        sigma = np.exp(z_logvar/2)
        return pred, mu, sigma, z

    def get_embedding(self, sess, observation):
        return sess.run(self.z, feed_dict={self.image: observation[None, :, :, :]})


class DiscreteAutoEncoder(BaseAutoEncoder):
    def __init__(self, *args, **kwargs):
        super(DiscreteAutoEncoder, self).__init__(*args, **kwargs)

        self.tau0 = 5.0
        self.tau_min = 0.01
        half_life = 5e5
        self.anneal_rate = np.log(2)/half_life
        self.tau = K.variable(self.tau_min, name="taur")
        tf.summary.scalar("hyper/tau", tf.reduce_mean(self.tau))

        self.KL_boost0 = 2.0
        self.KL_boost_min = 0.5  # TODO: Check value!
        # self.KL_boost_min = 5000  # TODO: Check value!
        # self.KL_boost_min = 1  # TODO: Check value!
        half_life = 1e6
        self.KL_boost_anneal_rate = np.log(2)/half_life
        self.KL_boost = K.variable(self.KL_boost_min, name="KL_boost_min")
        tf.summary.scalar("hyper/KL_boost", tf.reduce_mean(self.KL_boost))

        self.setup_network()

    def sample_z(self, q_y):
        N, M = self.latent_dim[0]  # Number variables, values per variable

        # # TODO: should it be logits or log(softmax(logits))? From the paper (Cat. reparam.) it looks like the latter!
        # U = K.random_uniform(K.shape(logits), 0, 1)
        # y = logits - K.log(-K.log(U + 1e-20) + 1e-20)  # logits + gumbel noise
        # y = K.reshape(y, (-1, self.N, self.M))

        # Gumbel softmax trick
        log_q_y = K.log(q_y + 1e-20)
        U = K.random_uniform(K.shape(log_q_y), 0, 1)
        y = log_q_y - K.log(-K.log(U + 1e-20) + 1e-20)  # log_prob + gumbel noise

        z = softmax(y / self.tau)
        z = K.reshape(z, (-1, N*M))
        return z


    def latent(self, x):
        print("Latent: Discrete")
        N, M = self.latent_dim[0]

        x = tf.layers.flatten(x)
        print(x)

        logits = tf.layers.dense(x, units=N*M, name='z_logits')
        logits = K.reshape(logits, (-1, N, M))
        print(logits)

        q_y = softmax(logits)
        print(q_y)

        z = self.sample_z(q_y)
        print(z)
        print()

        # tf.summary.image('logits', tf.expand_dims(logits, -1), self.tb_num_images)
        # tf.summary.image('q_y', tf.expand_dims(q_y, -1), self.tb_num_images)
        tf.summary.image('z', K.reshape(z, (-1, N, M, 1)), self.tb_num_images)

        tf.summary.histogram('train_D/logits', logits)
        tf.summary.histogram('train_D/q_y', q_y)
        tf.summary.histogram('train_D/z', z)

        return z, (logits, q_y)

    def compute_loss(self):
        N, M = self.latent_dim[0]
        _, q_y = self.latent_var

        rec_loss = self.reconstruction_loss()

        log_q_y = K.log(q_y + 1e-20)
        kl_loss = q_y * (log_q_y - K.log(1.0 / M))
        kl_loss = K.sum(kl_loss, axis=(1, 2))
        kl_loss *= self.KL_boost
        
        elbo = tf.reduce_mean(rec_loss + kl_loss)

        tf.summary.scalar("train/KL_loss", tf.reduce_mean(kl_loss))
        tf.summary.scalar("train/rec_loss", tf.reduce_mean(rec_loss))
        tf.summary.scalar("train/total_loss", elbo)
        return elbo

    def print_summary(self):
        print("tau {:5.2f}".format(K.get_value(self.tau)),
              "- KL_boost {:5.2f}".format(K.get_value(self.KL_boost)))

    def update_params(self, step):
        K.set_value(self.tau,
            np.max([self.tau_min, self.tau0 * np.exp(-self.anneal_rate * step)]))
        K.set_value(self.KL_boost, 
            np.max([self.KL_boost_min, self.KL_boost0 * np.exp(-self.KL_boost_anneal_rate * step)]))

    def get_embedding(self, sess, observation):
        raise NotImplementedError

    def predict(self, sess, data):
        raise NotImplementedError

