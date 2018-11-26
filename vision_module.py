import tensorflow as tf
import numpy as np
import os

from keras import backend as K
from keras.activations import softmax

from exp_parameters import DecayParam

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class BaseAutoEncoder(object):
    # Create model
    def __init__(self, exp_param):
        self.exp_param = exp_param
        self.dataset = exp_param.dataset
        self.latent_dim = exp_param.latent
        self.tb_num_images = 3

        self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.img_channels = exp_param.net_dim[-1]

        self.raw_input, self.image, self.mask_in, self.mask_net = self.create_net_input()
        tf.summary.image('image', self.image, self.tb_num_images)
        tf.summary.image('mask_net', self.mask_net, self.tb_num_images)

    def create_net_input(self):
        # tf.placeholder(tf.float32, (None,) + exp_param.data_dim, name='image')
        # self.image = tf.placeholder(tf.float32, (None,)+exp_param.input_dim, name='image')
        raw_input = tf.placeholder(self.exp_param.raw_type, (None,) + self.exp_param.raw_dim, name='raw_input')

        net_input = tf.image.resize_images(
            raw_input,
            size=self.exp_param.net_dim[:2],
            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        net_input = tf.cast(net_input, tf.float32)
        if self.dataset == 'breakout':
            net_input = tf.div(net_input, 255., 'normalize')

        mask_in = tf.placeholder(tf.uint8, (None,) + self.exp_param.raw_dim[:2] + (1,), 'Rec_loss_mask')
        mask_net = tf.image.resize_images(
            mask_in,
            size=self.exp_param.net_dim[:2],
            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        mask_net = tf.cast(mask_net, tf.float32)/255.
        if self.exp_param.g_size != 0:
            g_kernel = self.gaussian_kernel(self.exp_param.g_size, 0, self.exp_param.g_std)
            mask_net = tf.nn.conv2d(mask_net, g_kernel, strides=[1, 1, 1, 1], padding="SAME")
        mask_net = mask_net * self.exp_param.g_std / 0.3989  # https://stats.stackexchange.com/questions/143631/height-of-a-normal-distribution-curve
        mask_net += mask_net*self.exp_param.rec_loss_multiplier

        return raw_input, net_input, mask_in, mask_net


    def gaussian_kernel(self,
                        size: int,
                        mean: float,
                        std: float,):
        """ Makes 2D gaussian Kernel for convolution.
            https://stackoverflow.com/questions/52012657/how-to-make-a-2d-gaussian-filter-in-tensorflow
        """
        d = tf.distributions.Normal(float(mean), float(std))
        vals = d.prob(tf.range(start=-size, limit=size + 1, dtype=tf.float32))
        gauss_kernel = tf.einsum('i,j->ij', vals, vals)

        return (gauss_kernel / tf.reduce_sum(gauss_kernel))[:, :, tf.newaxis, tf.newaxis]

    def setup_network(self):
        self.encoder_out = self.encoder(self.image)
        self.z, self.latent_var = self.latent(self.encoder_out)
        self.reconstructions = self.decoder(self.z)
        tf.summary.image('reconstructions', self.reconstructions, self.tb_num_images)

        self.loss, self.loss_img = self.compute_loss()

        mask_norm = self.mask_net/(tf.reduce_max(self.mask_net)+1e-9)
        mask_norm = tf.tile(mask_norm, [1, 1, 1, 3])

        loss_img_3ch = self.loss_img/(tf.reduce_max(self.loss_img)+1e-9)
        loss_img_3ch = tf.tile(loss_img_3ch, [1, 1, 1, 3])

        sum_img_top = tf.concat([self.image, self.reconstructions], 2)
        sum_img_bot = tf.concat([mask_norm, loss_img_3ch], 2)
        self.sum_img = tf.concat([sum_img_top, sum_img_bot], 1)
        tf.summary.image('awesome_summary', self.sum_img, self.tb_num_images)

        self.merged = tf.summary.merge_all()

    def encoder(self, x):
        print('Encoder')
        print(x)
        if self.dataset == 'mnist':
            x = tf.layers.conv2d(x, filters=32, kernel_size=3, strides=2, padding='valid', activation=tf.nn.relu)
            print(x)
            x = tf.layers.conv2d(x, filters=64, kernel_size=3, strides=2, padding='valid', activation=tf.nn.relu)
            print(x)
            x = tf.layers.conv2d(x, filters=128, kernel_size=3, strides=2, padding='valid', activation=tf.nn.relu)
            print(x)
            print()
        elif self.dataset == 'breakout':
            x = tf.layers.conv2d(x, filters=8, kernel_size=3, strides=2, padding='same', activation=tf.nn.relu)
            print(x)
            x = tf.layers.conv2d(x, filters=16, kernel_size=3, strides=2, padding='same', activation=tf.nn.relu)
            print(x)
            x = tf.layers.conv2d(x, filters=32, kernel_size=3, strides=2, padding='same', activation=tf.nn.relu)
            print(x)
            x = tf.layers.conv2d(x, filters=64, kernel_size=3, strides=2, padding='same', activation=tf.nn.relu)
            print(x)
            x = tf.layers.conv2d(x, filters=128, kernel_size=3, strides=2, padding='same', activation=tf.nn.relu)
            print(x)
        return x

    def decoder(self, z):
        print('Decoder')

        first_conv_filters = 128
        decoder_input_size = self.encoder_out.shape[1]*self.encoder_out.shape[2]*first_conv_filters

        x = tf.layers.dense(z, decoder_input_size, activation=None)
        print(x)
        # x = tf.reshape(x, [-1, 1, 1, decoder_input_size])
        x = tf.reshape(x, [
            -1,
            self.encoder_out.shape[1],
            self.encoder_out.shape[2],
            first_conv_filters
        ])
        # x = tf.layers.conv2d(x, filters=128, kernel_size=3, strides=2, padding='same', activation=tf.nn.relu)
        print(x)

        x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=2, strides=2, padding='valid', activation=tf.nn.relu)
        # x = tf.layers.conv2d(x, filters=64, kernel_size=3, strides=2, padding='same', activation=tf.nn.relu)
        print(x)
        x = tf.layers.conv2d_transpose(x, filters=32, kernel_size=2, strides=2, padding='valid', activation=tf.nn.relu)
        # x = tf.layers.conv2d(x, filters=32, kernel_size=3, strides=2, padding='same', activation=tf.nn.relu)
        print(x)
        x = tf.layers.conv2d_transpose(x, filters=16, kernel_size=2, strides=2, padding='valid', activation=tf.nn.relu)
        # x = tf.layers.conv2d(x, filters=16, kernel_size=3, strides=2, padding='same', activation=tf.nn.relu)
        print(x)
        x = tf.layers.conv2d_transpose(x, filters=8, kernel_size=2, strides=2, padding='valid', activation=tf.nn.relu)
        # x = tf.layers.conv2d(x, filters=8, kernel_size=3, strides=2, padding='same', activation=tf.nn.relu)
        print(x)
        x = tf.layers.conv2d_transpose(x, filters=8, kernel_size=2, strides=2, padding='valid', activation=tf.nn.relu)
        # x = tf.layers.conv2d(x, filters=8, kernel_size=3, strides=2, padding='same', activation=tf.nn.relu)
        print(x)
        if self.dataset == 'mnist':
            x = tf.layers.conv2d(x, filters=self.img_channels, kernel_size=1, strides=1, padding='valid', activation=tf.nn.relu)
        elif self.dataset == 'breakout':
            x = tf.layers.conv2d(x, filters=self.img_channels, kernel_size=1, strides=1, padding='same', activation=tf.nn.relu)
        print(x)
        print()
        return x

    def sample_z(self, *args):
        raise NotImplementedError

    def latent(self, x):
        raise NotImplementedError

    def reconstruction_loss(self):
        # logits_flat = tf.layers.flatten(self.reconstructions)
        # labels_flat = tf.layers.flatten(self.image)
        # mask = tf.layers.flatten(self.mask)
        # err = logits_flat - labels_flat
        # err = err * mask

        err = (self.reconstructions - self.image)
        err = tf.square(err)
        err = err + err*self.mask_net

        return tf.reduce_mean(err, axis=[1, 2, 3]), tf.reduce_mean(err, axis=-1, keepdims=True)

    def KL_loss(self):
        raise NotImplementedError

    def compute_loss(self):
        raise NotImplementedError

    def update_params(self, *args, **kwargs):
        pass

    def get_embedding(self, sess, observation):
        raise NotImplementedError

    def predict(self, sess, data):
        raise NotImplementedError

    def print_summary(self):
        print()


class ContinuousAutoEncoder(BaseAutoEncoder):
    def __init__(self, *args, **kwargs):
        super(ContinuousAutoEncoder, self).__init__(*args, **kwargs)

        self.KL_boost = DecayParam(x0=0.01, x_min=0.001, half_life=5e5)  # TODO: should be handled by config
        tf.summary.scalar("hyper/KL_boost_C", tf.reduce_mean(self.KL_boost.value))

        self.setup_network()

    def sample_z(self, mu, logvar):
        eps = tf.random_normal(shape=tf.shape(mu))
        z = mu + tf.exp(logvar / 2) * eps
        return z

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

    def KL_loss(self):
        z_mu, z_logvar = self.latent_var
        kl_loss = 0.5 * tf.reduce_mean(tf.exp(z_logvar) + z_mu**2 - 1. - z_logvar, 1)
        return kl_loss*self.KL_boost.value

    def compute_loss(self):
        rec_loss, err_img = self.reconstruction_loss()
        kl_loss = self.KL_loss()
        vae_loss = tf.reduce_mean(rec_loss + kl_loss)

        tf.summary.scalar("train/KL_loss", tf.reduce_mean(kl_loss))
        tf.summary.scalar("train/rec_loss", tf.reduce_mean(rec_loss))
        tf.summary.scalar("train/total_loss", tf.reduce_mean(vae_loss))
        tf.summary.image('Error_image', err_img, self.tb_num_images)
        tf.summary.histogram('train_C/err_vals', err_img)
        return vae_loss, err_img

    def update_params(self, step):
        self.KL_boost.update_param(step)

    def get_embedding(self, sess, observation):
        return sess.run(self.z, feed_dict={self.image: observation[None, :, :, :]})

    def predict(self, sess, data):
        print(sess)
        z_mu, z_logvar = self.latent_var
        pred, mu, z_logvar, z = sess.run([self.reconstructions, z_mu, z_logvar, self.z], feed_dict={self.image:data})
        sigma = np.exp(z_logvar/2)
        return pred, mu, sigma, z

    def print_summary(self):
        print("KL_boost {:5.4f}".format(K.get_value(self.KL_boost.value)))


class DiscreteAutoEncoder(BaseAutoEncoder):
    def __init__(self, *args, **kwargs):
        super(DiscreteAutoEncoder, self).__init__(*args, **kwargs)

        self.tau = DecayParam(x0=5.0, x_min=0.01, half_life=7.5e5)  # TODO: should be handled by config
        tf.summary.scalar("hyper/tau", tf.reduce_mean(self.tau.value))

        self.KL_boost = DecayParam(x0=0.5, x_min=0.1, half_life=5e5)  # TODO: should be handled by config
        tf.summary.scalar("hyper/KL_boost_D", tf.reduce_mean(self.KL_boost.value))

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

        # z = K.reshape(softmax(y / self.tau), (-1, N*M))

        def gumble_softmax(y):
            print("gumble_softmax")
            z = softmax(y / self.tau.value)
            z = tf.reshape(z, (-1, N*M))
            return z

        def hardsample(log_q_y):
            print('hardsample')
            log_q_y = tf.reshape(log_q_y, (-1, M))
            z = tf.multinomial(log_q_y, 1)
            print("multinomial")
            print(z)
            z = tf.one_hot(z, M)
            print("onehot")
            print(z)
            z = tf.reshape(z, (-1, N*M))
            print("reshape")
            print(z)
            return z

        # TODO: make sure that hard sample works with differnet shapes
        z = tf.cond(
            self.is_training,
            lambda: gumble_softmax(y),
            lambda: hardsample(log_q_y)
        )

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

        # TODO: remove the 0th column.
        #z=
        print()

        # tf.summary.image('logits', tf.expand_dims(logits, -1), self.tb_num_images)
        # tf.summary.image('q_y', tf.expand_dims(q_y, -1), self.tb_num_images)
        tf.summary.image('z', K.reshape(z, (-1, N, M, 1)), self.tb_num_images)

        tf.summary.histogram('train_D/logits', logits)
        tf.summary.histogram('train_D/q_y', q_y)
        tf.summary.histogram('train_D/z', z)

        return z, (logits, q_y)

    def KL_loss(self):
        N, M = self.latent_dim[0]
        _, q_y = self.latent_var

        log_q_y = K.log(q_y + 1e-20)
        kl_loss = q_y * (log_q_y - K.log(1.0 / M))
        kl_loss = tf.reduce_mean(kl_loss, axis=(1, 2))
        return kl_loss*self.KL_boost.value

    def compute_loss(self):
        rec_loss, err_img = self.reconstruction_loss()
        kl_loss = self.KL_loss()
        elbo = tf.reduce_mean(rec_loss + kl_loss)

        tf.summary.scalar("train/KL_loss", tf.reduce_mean(kl_loss))
        tf.summary.scalar("train/rec_loss", tf.reduce_mean(rec_loss))
        tf.summary.scalar("train/total_loss", tf.reduce_mean(elbo))
        tf.summary.image('Error_image', err_img, self.tb_num_images)
        tf.summary.histogram('train_D/err_vals', err_img)
        return elbo, err_img

    def update_params(self, step):
        self.tau.update_param(step)
        self.KL_boost.update_param(step)

    def get_embedding(self, sess, observation):
        raise NotImplementedError

    def predict(self, sess, data):
        raise NotImplementedError

    def print_summary(self):
        print("tau {:5.4f}".format(K.get_value(self.tau.value)),
              "- KL_boost {:5.4f}".format(K.get_value(self.KL_boost.value)))
