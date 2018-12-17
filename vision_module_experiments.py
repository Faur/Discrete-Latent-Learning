import tensorflow as tf

from exp_parameters import ExpParam
from train_vision import train_vae

############### MNIST ###############


raw_dim = (28, 28, 1)
net_dim = (28, 28, 1)
sz = [2, 4, 8, 16, 32]

for j in range(5):
    latsz = sz[j]
    ## CONTINUOUS
    i = 1
    exp_param = ExpParam(
            lat_type="continuous",
            dataset='mnist',
            latent=[latsz],
            raw_type=tf.float32,
            raw_dim=raw_dim,
            net_dim=net_dim,
            # batch_size=16,
            max_example=1e5,
        )
    train_vae(exp_param, i)

    ## DISCRETE
    exp_param = ExpParam(
            lat_type="discrete",
            dataset='mnist',
            latent=[[latsz * 32, 2]],
            raw_type=tf.float32,
            raw_dim=raw_dim,
            net_dim=net_dim,
            learning_rate=0.001,
            # batch_size=16,
            max_example=1e5,
        )
    train_vae(exp_param, i)
    ## CONTINUOUS
    i = 2
    exp_param = ExpParam(
        lat_type="continuous",
        dataset='mnist',
        latent=[latsz],
        raw_type=tf.float32,
        raw_dim=raw_dim,
        net_dim=net_dim,
        # batch_size=16,
        max_example=1e5,
    )
    train_vae(exp_param, i)

    ## DISCRETE
    exp_param = ExpParam(
        lat_type="discrete",
        dataset='mnist',
        latent=[[latsz * 32, 2]],
        raw_type=tf.float32,
        raw_dim=raw_dim,
        net_dim=net_dim,
        learning_rate=0.001,
        # batch_size=16,
        max_example=1e5,
    )
    train_vae(exp_param, i)
print('Done')
