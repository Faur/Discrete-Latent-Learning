import tensorflow as tf
import numpy as np
import time

import data_utils
from vision_module import ContinuousAutoEncoder, DiscreteAutoEncoder

np.random.seed(int(time.time()))
tf.set_random_seed(int(time.time()))

# TODO: Things to consider
# * Use numpy for scalar tracking?
# * Rec loss should be independeont of number of pixels
# * KL loss should be independent of number of dimensions

# TODO: Handle this better!

def create_or_load_vae(model_path, network_args):
    graph = tf.Graph()
    with graph.as_default():  # Original formuation
        # graph.as_default()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        # sess = tf.Session(config=config, graph=graph) # previous
        sess = tf.InteractiveSession(config=config, graph=graph)

        if "continuous" in model_path:
            print("Continuous")
            network = ContinuousAutoEncoder(network_args)
        elif 'discrete' in model_path:
            print("Discrete")
            network = DiscreteAutoEncoder(network_args)
        else:
            print("Undefined")
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


def train_vae(AE_type):
    ### GENERAL SETUP
    experiment_name = AE_type +"_"+ str(time.time())
    model_path = "saved_model_" + AE_type + "/"
    model_name = model_path + experiment_name + '_model'

    print('experiment_name', experiment_name)
    print('model_path', model_path)
    print('model_name', model_name)
    print()

    if 'continuous' in experiment_name:
        network_args = [32]
    elif 'discrete' in experiment_name:
        network_args = [[32, 16]]
    else:
        raise Exception

    ### DATA
    batch_size = 4  # TODO: Use real
    train_iter, test_iter = data_utils.load_data(batch_size, 'mnist')

    ### NETWORK
    sess, network, saver = create_or_load_vae(model_path, network_args=network_args)

    # TODO: load or inferr gloabl step (don't start at zero!)
    global_step = tf.Variable(0, name='global_step', trainable=False)

    train_op = tf.train.AdamOptimizer(0.001).minimize(network.loss, global_step=global_step)
    tf.global_variables_initializer().run()

    writer = tf.summary.FileWriter('logdir/'+experiment_name)
    step = global_step.eval()

    print("\nBegin Training")
    try:
        while True:
            epoch, e_step, images = next(train_iter)

            _, loss_value = sess.run([train_op, network.loss],
                                feed_dict={network.image: images})
            network.update_params(e_step)

            if np.isnan(loss_value):
                raise ValueError('Loss value is NaN')
            if step % 10 == 0 and step > 0:
                print("Epoch {:5}, obs {:9}: T. loss {:9.3f}".format(
                    epoch, e_step, loss_value), end=' ### ')
                network.print_summary()

                [summary] = sess.run([network.merged], feed_dict={network.image: images})
                writer.add_summary(summary, step*batch_size)
                try:
                    save_path = saver.save(sess, model_name, global_step=global_step)
                except:
                    print("\nFAILED TO SAVE MODEL!\n")

            step += 1

    except (KeyboardInterrupt, SystemExit):
        print("Manual Interrupt")

    except Exception as e:
        print("Exception: {}".format(e))


if __name__ == '__main__':
    AE_types = ["continuous", "discrete"]
    # TODO: Beter switching logic handling!
    # train_vae(AE_types[0])
    train_vae(AE_types[1])

