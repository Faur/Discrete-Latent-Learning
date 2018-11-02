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
# * Add weight decay / monitor weights
# * Validation should use hard samples!
# * save best validation score model
# * include a 'degree of determinism' measure in tensorboard.
# * Exponential smoothing should be towards a point, not just end abruptly.
# 		Parameters: What should it annealt towards? and how many steps before it is 1% away from that?


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


def train_vae(AE_type, network_args, experiment_name=None):
    ### GENERAL SETUP
    if experiment_name is None:
    	experiment_name = AE_type +"_"+ str(time.time())
    model_path = "saved_model_" + AE_type + "/"
    model_name = model_path + experiment_name + '_model'

    print('experiment_name', experiment_name)
    print('model_path', model_path)
    print('model_name', model_name)
    print()

    ################## SETTINGS #####################
    batch_size = 64  # TODO: Use real
    learning_rate = 0.001
    ### DATA
    train_iter, test_iter = data_utils.load_data(batch_size, 'mnist')

    ### NETWORK
    sess, network, saver = create_or_load_vae(model_path, network_args=network_args)

    # TODO: load or inferr gloabl step (don't start at zero!)
    global_step = tf.Variable(0, name='global_step', trainable=False)

    train_op = tf.train.AdamOptimizer(learning_rate).minimize(network.loss, global_step=global_step)
    tf.global_variables_initializer().run()

    writer = tf.summary.FileWriter('logdir/'+experiment_name)
    writer_test = tf.summary.FileWriter('logdir/'+experiment_name+'_test')
    step = global_step.eval()

    print("\nBegin Training")
    try:
        while True:
            epoch, e_step, images = next(train_iter)

            _, loss_value = sess.run([train_op, network.loss],
                                feed_dict={network.image: images})
            loss_value = np.mean(loss_value)
            network.update_params(step*batch_size)

            if np.any(np.isnan(loss_value)):
                raise ValueError('Loss value is NaN')

            valid_inter = 100
            if step % (valid_inter*10) == 0:
                _, _, images = next(test_iter)
                
                # TODO: Test should use hard sample
                [summary, test_loss] = sess.run([network.merged, network.loss], 
                    feed_dict={network.image: images})
                test_loss = np.mean(test_loss)
                writer_test.add_summary(summary, step*batch_size)

                print("Epoch {:5}, obs {:12}: Te. loss {:9.3f}".format(
                    epoch, step*batch_size, test_loss), end=' ### ')
                network.print_summary()
                print()

            if step % valid_inter == 0 and step > 0:
                print("Epoch {:5}, obs {:12}: Tr. loss {:9.3f}".format(
                    epoch, step*batch_size, loss_value), end=' ### ')
                network.print_summary()

                [summary] = sess.run([network.merged], feed_dict={network.image: images})
                writer.add_summary(summary, step*batch_size)
                try:
                    save_path = saver.save(sess, model_name, global_step=global_step)
                except KeyboardInterrupt:
                    break
                except:
                    print("\nFAILED TO SAVE MODEL!\n")

            step += 1

            if epoch >= 100:
            	print("Max epoch reached!")
            	break

    except (KeyboardInterrupt, SystemExit):
        print("Manual Interrupt")

    except Exception as e:
        print("Exception: {}".format(e))


if __name__ == '__main__':
    AE_types = ["continuous", "discrete"]
    # TODO: Beter switching logic handling!

    network_args = [1]
    train_vae(AE_types[0], network_args, 'continuous_1_' + str(time.time())) 

    network_args = [2]
    train_vae(AE_types[0], network_args, 'continuous_2_' + str(time.time())) 

    network_args = [4]
    train_vae(AE_types[0], network_args, 'continuous_4_' + str(time.time())) 

    # network_args = [8]
    # train_vae(AE_types[0], network_args, 'continuous_8_' + str(time.time())) 



    # network_args = [[2, 2]]
    # train_vae(AE_types[1], network_args, 'disc_2_2_' + str(time.time()))

    # network_args = [[4, 2]]
    # train_vae(AE_types[1], network_args, 'disc_4_2_' + str(time.time()))

    # network_args = [[8, 2]]
    # train_vae(AE_types[1], network_args, 'disc_8_2_' + str(time.time()))

    network_args = [[16, 2]]
    train_vae(AE_types[1], network_args, 'disc_16_2_' + str(time.time()))

    network_args = [[32, 2]]
    train_vae(AE_types[1], network_args, 'disc_32_2_' + str(time.time()))

    network_args = [[64, 2]]
    train_vae(AE_types[1], network_args, 'disc_64_2_' + str(time.time()))

    network_args = [[128, 2]]
    train_vae(AE_types[1], network_args, 'disc_128_2_' + str(time.time()))

    # network_args = [[256, 2]]
    # train_vae(AE_types[1], network_args, 'disc_256_2_' + str(time.time()))
