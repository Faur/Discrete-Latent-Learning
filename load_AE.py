import tensorflow as tf
import numpy as np
import pickle

from exp_parameters import ExpParam
from train_vision import create_or_load_vae
from gen_raw_data import __observation_update

from A2C.ActorCritic import A2C
from A2C.utils.utils import parse_args, create_experiment_dirs


def encode_data(AE, sess, obs):
    obs_mask = np.zeros_like(obs)
    [z] = sess.run([AE.z], feed_dict={
        AE.raw_input: obs,
        AE.mask_in: obs_mask,
        AE.is_training: False
    })

    if z.shape[-1] == 8192:
        z = np.reshape(z, [-1, 4096, 2])[:, :, -1]

    return z


def main():
    model_name = 'breakout_discrete_BLM64_STD0_lr0.0001_LAT4096(2)_MADE1543847099'
    model_path = 'C:\\Users\\Toke\\Dropbox\\MAI\\'

    model_name = 'breakout_discrete_BLM1_STD0_LAT4096(2)_MADE1544445274'
    model_path = 'C:\\Users\\Toke\\Git\\Discrete-Latent-Learning\\saved_model\\'

    model_path += model_name

    latent = [[32 * 128, 2]]
    # raw_dim = (210, 160, 3)
    # net_dim = (32*4, 32*3, 3)
    raw_dim = (84, 84, 4)
    net_dim = (84, 84, 4)

    ### Do stuff
    exp_param = ExpParam(
        lat_type="discrete",
        dataset='breakout',
        latent=[[32 * 128, 2]],
        raw_type=tf.uint8,
        raw_dim=raw_dim,
        net_dim=net_dim,  # very close to org aspect ration
        batch_size=2,  # for testing
    )

    ### Load model
    sess_ae, AE, saver = create_or_load_vae(
        model_path,
        exp_param=exp_param,
        critical_load=True)


    graph_a2c = tf.Graph()
    with graph_a2c.as_default():
        # tf.reset_default_graph()

        config_args = parse_args()
        config = tf.ConfigProto(allow_soft_placement=True,
                                intra_op_parallelism_threads=config_args.num_envs,
                                inter_op_parallelism_threads=config_args.num_envs)

        config.gpu_options.allow_growth = True
        sess_a2c = tf.Session(config=config)

        config_args.experiment_dir, config_args.summary_dir, config_args.checkpoint_dir, config_args.output_dir, config_args.test_dir = \
            create_experiment_dirs(config_args.experiment_dir)

        a2c = A2C(sess_a2c, config_args)

        # testing
        with open(a2c.args.experiment_dir + a2c.args.env_name + '.pkl', 'rb') as f:
            observation_space_shape, action_space_n = pickle.load(f)

        env = a2c.make_all_environments(
            num_envs=1,
            env_class=a2c.env_class,
            env_name=a2c.args.env_name,
            seed=a2c.args.env_seed)

        a2c.model.build(observation_space_shape, action_space_n)

        a2c.trainer._init_model()
        a2c.trainer._load_model()

        states = a2c.trainer.model.step_policy.initial_state

        dones = [False for _ in range(env.num_envs)]

        observation_s = np.zeros(
            (env.num_envs, a2c.trainer.model.img_height, a2c.trainer.model.img_width,
             a2c.trainer.model.num_classes * a2c.trainer.model.num_stack),
            dtype=np.uint8)

        observation = env.reset()
        observation_s = __observation_update(observation, observation_s)

        i = 0
        max_steps = 1e3
        while i < max_steps:
            i += 1
            observation_z = encode_data(AE, sess_ae, observation_s)

            ## TODO: Change a2c.model.step_policy.step
            actions, values, states = a2c.model.step_policy.step(observation_z, states, dones)

            observation, rewards, dones, _ = env.step(actions)

            for n, done in enumerate(dones):
                if done:
                    observation_s[n] *= 0
                    # print(file_name, i, len(observation_list), max_steps, end='\r')
                    # print(batch_num, len(observation_list), max_steps)

            print()
            env.render()


if __name__ == '__main__':
    main()


