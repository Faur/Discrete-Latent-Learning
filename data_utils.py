import numpy as np
import tensorflow as tf

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

