import numpy as np
import multiprocessing as mp
import gym
import gym.spaces #TODO: remove this (only used to suppress warning

import gym_utils
import data_utils


def generate_action(env):
    a = env.action_space.sample()
    return a


def gen_data(gen_args, render=False):
    """ Format: (obs, action, reward, done)
    """
    batch_num, postfix, max_steps = gen_args

    env = gym.make("Breakout-v0")
    obs_data = []

    obs, reward_sum, done = gym_utils.reset_env(env)
    obs = data_utils.normalize_observation(obs)

    for i in range(max_steps):
        action = generate_action(env)
        obs_, reward, done, info = env.step(action)
        obs_data.append((obs, action, reward, done))

        if render: env.render()

        if done:
            obs, reward_sum, done = gym_utils.reset_env(env)
            obs = data_utils.normalize_observation(obs)

        else:
            obs = data_utils.normalize_observation(obs_)

    # data_as_array = np.concatenate(obs_data, 0)
    data_as_array = np.vstack(obs_data)

    ### Compute memory usage of obs
    # size_of_data = data_utils.getSize_lol(obs_data)
    # actual_total_obs = len(obs_data[0]) * len(obs_data)
    # size_per_obs = int(size_of_data/actual_total_obs)
    # print(data_utils.sizeof_fmt(size_per_obs))  # 24.6KiB
    # print(size_per_obs)  # 25218 Bytes

    file_name = 'Breakout_raw_{}_{:04d}'.format(postfix, batch_num)
    data_utils.save_np_array_as_h5(file_name, data_as_array)
    # print('Generated dataset with ', data_as_array.shape[0], "observations.")
    # print("Format: (obs, action, reward, done)")
    print('Saved batch: {:4}'.format(batch_num), '-', file_name)

    env.close()
    # return obs_data
    return file_name


def generate_raw_data(total_frames, postfix=''):
    total_frames = int(total_frames)

    # 256*32*obs_mem_size ~ 0.75 GB
    max_eps_len = 256  # doesn't actually matter!
    max_frames_per_thread = max_eps_len*32
    num_batches = total_frames // max_frames_per_thread
    frames_in_last_batch = total_frames - max_frames_per_thread * (total_frames // max_frames_per_thread)
    batch_len = [max_frames_per_thread]*num_batches + [frames_in_last_batch]
    if frames_in_last_batch != 0:
        num_batches += 1
    gen_args = [(i, postfix, batch_len[i]) for i in range(num_batches)]

    print("Generating", postfix, "data for env CarRacing-v0")
    print("total_frames: ", total_frames)
    print('max_frames_per_thread', max_frames_per_thread)
    print("num_batches:", num_batches, '(',batch_len,')')
    print('...')

    with mp.Pool(mp.cpu_count()-1) as p:
        # data = p.map(gen_data, gen_args)
        file_names = p.map(gen_data, gen_args)

    return file_names


if __name__ == '__main__':
    # TODO: This causes memory issues!
    train_frames = 1e4
    train_frames = 1e6
    file_names = generate_raw_data(train_frames, 'train')
    print('\n'*2)

    valid_frames = 1e4
    valid_frames = 1e5
    file_names = generate_raw_data(valid_frames, 'valid')

    if 0:
        print("Load test - Begin.")
        file_name = file_names[0]
        data_path = './data/' + file_name + '.h5'
        print(data_path)
        data = data_utils.lad_h5_as_np_array(data_path)
        print('data', type(data))
        print("Load test - Success!")






