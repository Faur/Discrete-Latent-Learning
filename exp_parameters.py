import time
import numpy as np
from keras import backend as K


class DecayParam():
    def __init__(self, x0, x_min, half_life, name=None):

        self.x0 = x0
        self.x_min = x_min
        self.half_life = half_life
        self.anneal_rate = np.log(2)/self.half_life
        self.value = K.variable(self.x_min, name=name)

    def update_param(self, step):
        K.set_value(self.value, np.max([self.x_min, self.x0 * np.exp(-self.anneal_rate * step)]))


class ExpParam():
    def __init__(self,
                 lat_type,
                 latent,
                 dataset,
                 input_dim,
                 data_dim=None,
                 learning_rate=0.001,
                 valid_inter=100,
                 batch_size=64,
                 ):

        self.created = str(int(time.time()))
        assert lat_type in ["continuous", "discrete"], 'lat_type, ' + str(lat_type) + ' not understood.'
        self.lat_type = lat_type
        self.latent = latent

        assert dataset in ['mnist'], 'dataset, ' + str(dataset) + ' not understood.'
        self.dataset = dataset
        self.input_dim = input_dim  # the input to the newtork
        self.data_dim = data_dim if data_dim is not None else input_dim  # the raw input data

        self.learning_rate = learning_rate
        self.valid_inter = valid_inter
        self.batch_size = batch_size

    def toString(self):
        out = self.dataset + '_' + self.lat_type

        out += '_LAT'
        if self.lat_type == 'discrete':
            for dim in self.latent:
                out += str(dim[0]) + '(' + str(dim[1]) + ')'
        elif self.lat_type == 'continuous':
            for dim in self.latent:
                out = out + str(dim)

        out += '_MADE' + self.created

        return out

    def copy(self):
        # TODO!
        raise NotImplementedError

    def save(self):
        # TODO!
        raise NotImplementedError

    def load(self):
        # TODO!
        raise NotImplementedError

    def print(self):
        # TODO: Actually print all the params in a reasonable way
        print(self.toString())


