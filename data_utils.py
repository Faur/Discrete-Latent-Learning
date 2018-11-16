import h5py
import numpy as np
import sys
import os

import tensorflow as tf


def data_iterator_mnist(data, batch_size):
    N = data.shape[0]
    epoch = 0
    while True:
        epoch += 1
        if batch_size == -1:
            yield None, N, data
            
        np.random.shuffle(data)
        for i in range(int(N/batch_size)):
            yield epoch, i*batch_size, data[i*batch_size:(i+1)*batch_size]


def data_iterator_atari(data, batch_size):
    obs, action, reward, done = data
    N = obs.shape[0]
    epoch = 0
    while True:
        epoch += 1
        if batch_size == -1:
            yield None, N, (obs, action, reward, done)

        np.random.shuffle(data)
        for i in range(int(N / batch_size)):
            out_data = (
                obs[i * batch_size:(i + 1) * batch_size],
                action[i * batch_size:(i + 1) * batch_size],
                reward[i * batch_size:(i + 1) * batch_size],
                done[i * batch_size:(i + 1) * batch_size],
            )
            yield epoch, i * batch_size, out_data


def load_data(train_batch_size, dataset='mnist', test_batch_size=-1):
    if dataset == 'mnist':
        mnist = tf.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = np.expand_dims(x_train/255., -1)
        x_test = np.expand_dims(x_test/255., -1)

        train_iter = data_iterator_mnist(x_train, batch_size=(train_batch_size))
        test_iter = data_iterator_mnist(x_test, batch_size=test_batch_size)

        return train_iter, test_iter
    elif dataset == 'breakout':
        # TODO: This probably causes meomry issues
        x_train = lad_h5_as_array('Breakout_raw_train_')
        train_iter = data_iterator_atari(x_train, batch_size=(train_batch_size))

        x_test = lad_h5_as_array('Breakout_raw_valid_')
        test_iter = data_iterator_atari(x_test, batch_size=(test_batch_size))

        return train_iter, test_iter
    else:
        print(dataset)
        raise NotImplementedError


def normalize_observation(observation):
    # obs = np.copy(observation)/255. # uses a lot more space!
    obs = np.copy(observation)
    return obs


def save_np_array_as_h5(file_name, data_as_array):
    # print("Format: (obs, action, reward, done)")
    data_path = './data/'+file_name+'.h5'
    # print("Saving dataset at: {}".format(data_path), end=' ... ')

    h5f = h5py.File(data_path, 'w')
    h5f.create_dataset('obs',    data=np.array([i for i in data_as_array[:, 0]]))
    h5f.create_dataset('action', data=data_as_array[:, 1].astype(int))
    h5f.create_dataset('reward', data=data_as_array[:, 2].astype(float))
    h5f.create_dataset('done',   data=data_as_array[:, 3].astype(int))
    h5f.close()


def lad_h5_as_list(data_path):
    h5f = h5py.File(data_path, 'r')
    # data = {}
    # data['obs']    = h5f['obs'][:]      # float
    # data['action'] = h5f['action'][:]   # int
    # data['reward'] = h5f['reward'][:]   # float
    # data['done']   = h5f['done'][:]     # int
    data = [
        h5f['obs'][:],
        h5f['action'][:],   # int
        h5f['reward'][:],   # float
        h5f['done'][:],     # int
        ]
    h5f.close()

    return data


def lad_h5_as_array(file_name, num_chars=4):
    data = []
    i = 0
    while True:
        data_path = './data/' + file_name + '{:04}'.format(i) + '.h5'

        if os.path.isfile(data_path):
            data.append(lad_h5_as_list(data_path))
        else:
            break
        print('Loaded:', data_path)
        i += 1

    # print("Format: (obs, action, reward, done)")
    obs = np.vstack([data[i][0] for i in range(len(data))])
    action = np.concatenate([data[i][1] for i in range(len(data))])
    reward = np.concatenate([data[i][2] for i in range(len(data))])
    done = np.concatenate([data[i][3] for i in range(len(data))])

    return obs, action, reward, done

def getSize_lol(lol):
    """ Get size from list of list of objects"""
    size = 0
    # for ll in lol:  # runs
    #     for l in ll:  # observations
    for l in lol:  # observations
        for t in l:  # individual elements
                if type(t) is np.ndarray:
                    size += t.nbytes
                else:
                    size += sys.getsizeof(t)

    return size


def sizeof_fmt(num, suffix='B'):
    """ From: https://stackoverflow.com/questions/1094841/reusable-library-to-get-human-readable-version-of-file-size/1094933#1094933"""
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)