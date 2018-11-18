import tensorflow as tf
import numpy as np
import time

import data_utils
from exp_parameters import ExpParam
from vision_module import ContinuousAutoEncoder, DiscreteAutoEncoder

np.random.seed(int(time.time()))
tf.set_random_seed(int(time.time()))

# TODO: Things to consider
# * Do experiments on MNIST
# * Try making KL_boost = 0. Companre + KL should perhaps be increasing, rather than decreasing?
# * Consider more powerful decoder

# * Add weight decay / monitor weights
# * save best validation score model

# * Don't do the one-hot encoding.

# * Use numpy for scalar tracking?
# * include a 'degree of determinism' measure in tensorboard.
# * Exponential smoothing should be towards a point, not just end abruptly.
#         Parameters: What should it annealt towards? and how many steps before it is 1% away from that?

# * Tensorboard measure: difference between two random prediction - pure prior check.
#   - This can be achieved by looking at q_y

def create_or_load_vae(model_path, exp_param):
    graph = tf.Graph()
    with graph.as_default():  # Original formuation
        # graph.as_default()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        # sess = tf.Session(config=config, graph=graph) # previous
        sess = tf.InteractiveSession(config=config, graph=graph)

        if "continuous" == exp_param.lat_type:
            print("Continuous")
            network = ContinuousAutoEncoder(exp_param)
        elif 'discrete' == exp_param.lat_type:
            print("Discrete")
            network = DiscreteAutoEncoder(exp_param)
        else:
            print("Undefined", model_path)
            raise NotImplementedError

        init = tf.global_variables_initializer()
        sess.run(init)

        saver = tf.train.Saver(max_to_keep=1)
        try:
            saver.restore(sess, tf.train.latest_checkpoint(model_path))
            print("Model restored from: {}".format(model_path))
        except:
            print("Could not restore saved model")

        return sess, network, saver


def train_vae(exp_param, experiment_name=None):
    ### GENERAL SETUP
    if experiment_name is None:
        experiment_name = exp_param.toString()
    model_path = "saved_model_" + experiment_name + "/"
    model_name = model_path + experiment_name + '_model'

    print('experiment_name: ', experiment_name)
    print('model_path: ', model_path)
    print('model_name: ', model_name)
    exp_param.print()
    print()

    ################## SETTINGS #####################
    valid_inter = exp_param.valid_inter
    batch_size = exp_param.batch_size
    learning_rate = exp_param.learning_rate
    data_set = exp_param.dataset

    ### DATA
    train_iter, test_iter = data_utils.load_data(batch_size, data_set)
    ball_col = data_utils.ball_col
    ball_loss_multiplier = exp_param.ball_loss_multiplier

    ### NETWORK
    sess, network, saver = create_or_load_vae(model_path, exp_param=exp_param)

    # TODO: load or inferr gloabl step (don't start at zero!)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    epoch = -1

    train_op = tf.train.AdamOptimizer(learning_rate).minimize(network.loss, global_step=global_step)
    tf.global_variables_initializer().run()

    writer = tf.summary.FileWriter('logdir/'+experiment_name)
    writer_test = tf.summary.FileWriter('logdir/'+experiment_name+'_test')
    step = global_step.eval()

    print("\nBegin Training")
    try:
        while True:
            network.update_params(step*batch_size)

            ## PERFORM TEST SET EVALUATION
            if step % (valid_inter*10) == 0: 
                _, _, images = next(test_iter)

                if exp_param.dataset == 'breakout':
                    # TODO: handle data for agent properly!
                    images = images[0]
                    masks = data_utils.mask_col(images, ball_col, ball_loss_multiplier)

                # TODO: Test should use hard sample
                [summary, test_loss] = sess.run([network.merged, network.loss], feed_dict={
                    network.raw_input: images,
                    network.mask: masks,
                    network.is_training: False
                })
                test_loss = np.mean(test_loss)
                writer_test.add_summary(summary, step*batch_size)

                print("Epoch {:5}, obs {:12}: Te. loss {:9.3f}".format(
                    epoch, step*batch_size, test_loss), end=' ### ')
                network.print_summary()
                print()

            epoch, e_step, images = next(train_iter)
            if exp_param.dataset == 'breakout':
                # TODO: handle data for agent properly!
                images = images[0]
                masks = data_utils.mask_col(images, ball_col, ball_loss_multiplier)

            ## COMPUTE TRAIN SET SUMMARY
            if step % valid_inter == 0 and step > 0:
                print("Epoch {:5}, obs {:12}: Tr. loss {:9.3f}".format(
                    epoch, step*batch_size, loss_value), end=' ### ')
                network.print_summary()

                [summary] = sess.run([network.merged], feed_dict={
                    network.raw_input: images,
                    network.mask: masks,
                    network.is_training: True
                })
                writer.add_summary(summary, step*batch_size)
                try:
                    save_path = saver.save(sess, model_name, global_step=global_step)
                except KeyboardInterrupt:
                    break
                except:
                    print("\nFAILED TO SAVE MODEL!\n")

            ## PERFORM TRAINING STEP
            _, loss_value = sess.run([train_op, network.loss], feed_dict={
                network.raw_input: images,
                network.mask: masks,
                network.is_training: True
                })
            loss_value = np.mean(loss_value)

            if np.any(np.isnan(loss_value)):
                raise ValueError('Loss value is NaN')

            if epoch >= exp_param.max_epoch:
                print("Max epoch reached!")
                break

            step += 1

    except (KeyboardInterrupt, SystemExit):
        print("Manual Interrupt")

    # except Exception as e:
    #     print("Exception: {}".format(e))
    sess.close()


if __name__ == '__main__':

    raw_dim = (210, 160, 3)
    net_dim = (32*4, 32*3, 3)
    exp_param = ExpParam(
        lat_type="continuous",
        dataset='breakout',
        latent=[8],
        raw_type=tf.float32,
        raw_dim=raw_dim,
        net_dim=net_dim,  # very close to org aspect ration
        learning_rate=0.001,
#        batch_size=2,  # for testing
    )
    train_vae(exp_param)

    ## CONTINUOUS
    # raw_dim = (28, 28, 1)
    # net_dim = (28, 28, 1)
    # exp_param = ExpParam(
    #     lat_type="continuous",
    #     dataset='mnist',
    #     latent=[2],
    #     raw_type=tf.float32,
    #     raw_dim=raw_dim,
    #     net_dim=net_dim,
    #     learning_rate=0.001,
    #     # batch_size=2,  # for testing
    # )
    # train_vae(exp_param)
    #
    # ## DISCRETE
    # exp_param = ExpParam(
    #     lat_type="discrete",
    #     dataset='mnist',
    #     latent=[[2, 2]],
    #     input_dim=(28, 28, 1),
    #     data_dim=(28, 28, 1),
    #     learning_rate=0.001,
    #     # batch_size=2,  # for testing
    # )
    # train_vae(exp_param)

    print('Done')