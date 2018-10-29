import tensorflow as tf
import numpy as np

import data_utils
from vision_module import ContinuousAutoEncoder

# np.random.seed(0)
# tf.set_random_seed(0)

# TODO: Handle this better!

def create_or_load_vae(model_path):
    graph = tf.Graph()
    # with graph.as_default():  # Original formuation
    graph.as_default()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # sess = tf.Session(config=config, graph=graph) # previous
    sess = tf.InteractiveSession(config=config)

    if "continuous" in model_path:
        print("Continuous")
        network = ContinuousAutoEncoder([32])
    elif 'discrete' in model_path:
        print("Discrete")
        raise NotImplementedError
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


def train_vae(experiment_name):
    model_path = "saved_model_" + experiment_name + "/"
    model_name = model_path + '_model'

    ### DATA
    batch_size = 2  # TODO: Use real
    train_iter, test_iter = data_utils.load_data(batch_size, 'mnist')

    ### NETWORK
    sess, network, saver = create_or_load_vae(model_path)

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

            if np.isnan(loss_value):
                raise ValueError('Loss value is NaN')
            if step % 10 == 0 and step > 0:
                print("Epoch {:5}, obs {:9}: training loss {:.3f}".format(
                    epoch, e_step*batch_size, loss_value))
                [summary] = sess.run([network.merged], feed_dict={network.image: images})
                writer.add_summary(summary, step*batch_size)
                save_path = saver.save(sess, model_name, global_step=global_step)

            step += 1

    except (KeyboardInterrupt, SystemExit):
        print("Manual Interrupt")

    except Exception as e:
        print("Exception: {}".format(e))


if __name__ == '__main__':
    AE_types = ["continuous", "discrete"]
    train_vae(AE_types[0])
    train_vae(AE_types[1])

