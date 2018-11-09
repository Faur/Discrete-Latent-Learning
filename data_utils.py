import h5py
import numpy as np
import tensorflow as tf
import sys

def data_iterator(data, batch_size):
    N = data.shape[0]
    epoch = 0
    while True:
        epoch += 1
        if batch_size == -1:
            yield None, data.shape[0], data
            
        np.random.shuffle(data)
        for i in range(int(N/batch_size)):
            yield epoch, i*batch_size, data[i*batch_size:(i+1)*batch_size]


def load_data(train_batch_size, dataset='mnist', test_batch_size=-1):
    if dataset == 'mnist':
        mnist = tf.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = np.expand_dims(x_train/255., -1)
        x_test = np.expand_dims(x_test/255., -1)

        train_iter = data_iterator(x_train, batch_size=(train_batch_size))
        test_iter = data_iterator(x_test, batch_size=test_batch_size)

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
    h5f.create_dataset('obs',    data=data_as_array[:, 0][0])
    h5f.create_dataset('action', data=data_as_array[:, 1].astype(int))
    h5f.create_dataset('reward', data=data_as_array[:, 2].astype(float))
    h5f.create_dataset('done',   data=data_as_array[:, 3].astype(int))
    h5f.close()


def lad_h5_as_np_array(data_path):
    h5f = h5py.File(data_path, 'r')
    data = {}
    data['obs']    = h5f['obs'][:]      # float
    data['action'] = h5f['action'][:]   # int
    data['reward'] = h5f['reward'][:]   # float
    data['done']   = h5f['done'][:]     # int
    h5f.close()
    return data


def getSize_lol(lol):
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