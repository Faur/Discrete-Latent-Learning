import numpy as np
import multiprocessing as mp
import gym
import gym.spaces #TODO: remove this (only used to suppress warning

import scipy.ndimage
import matplotlib.pyplot as plt

import gym_utils
import data_utils


def generate_action(env):
    a = env.action_space.sample()
    return a


def gen_data(gen_args, render=False):
    """ Format: (obs, action, reward, done)
    """
    batch_num, postfix, max_steps, frame_skip = gen_args

    env = gym.make("Breakout-v0")
    obs_data = []

    obs, reward_sum, done = gym_utils.reset_env(env)
    obs = data_utils.normalize_observation(obs)

    i = 0
    while len(obs_data) < max_steps:
        action = generate_action(env)
        obs_, reward, done, info = env.step(action)

        obs_mask = obs_.astype(int) - obs
        # obs_mask = np.abs(obs_mask)
        obs_mask = obs_mask * (obs_mask < 0)
        obs_mask = np.mean(obs_mask, -1, keepdims=True).astype(np.uint8)
        # obs_mask = obs_mask / 255.
        # obs_mask = scipy.ndimage.filters.gaussian_filter(obs_mask, 5)
        # plt.imshow(obs_mask[:, :, 0]); plt.figure(); plt.imshow(obs); plt.show()

        if i % frame_skip == 0:
            obs_data.append((obs, obs_mask, action, reward, done))

            if render: env.render()

        if done:
            obs, reward_sum, done = gym_utils.reset_env(env)
            obs = data_utils.normalize_observation(obs)
        else:
            obs = data_utils.normalize_observation(obs_)
        i += 1

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
    # print("Format: (obs, obs_mask, action, reward, done)")
    print('Saved batch: {:4}'.format(batch_num), '-', file_name)

    env.close()
    # return obs_data
    return file_name


def generate_raw_data(total_frames, postfix='', frame_skip=1):
    total_frames = int(total_frames)

    # 256*32*obs_mem_size ~ 0.75 GB
    max_eps_len = 512  # doesn't actually matter - just max file size thing
    max_frames_per_thread = max_eps_len*32
    num_batches = total_frames // max_frames_per_thread
    frames_in_last_batch = total_frames - max_frames_per_thread * (total_frames // max_frames_per_thread)
    batch_len = [max_frames_per_thread]*num_batches + [frames_in_last_batch]
    if frames_in_last_batch != 0:
        num_batches += 1
    gen_args = [(i, postfix, batch_len[i], frame_skip) for i in range(num_batches)]

    num_threads = mp.cpu_count()-1
    print("Generating", postfix, "data for env CarRacing-v0")
    print("total_frames: ", total_frames)
    print('max_frames_per_thread', max_frames_per_thread)
    print("num_batches:", num_batches, '(',batch_len,')')
    print("num_threads:", num_threads)
    print('...')

    with mp.Pool(num_threads) as p:
        # data = p.map(gen_data, gen_args)
        file_names = p.map(gen_data, gen_args)

    return file_names


if __name__ == '__main__':
    frame_skip = 4

    # TODO: This causes memory issues!
    train_frames = 5e4
    train_frames = 1e4
    # train_frames = 1e2
    file_names = generate_raw_data(train_frames, 'train', frame_skip)
    print('\n'*2)

    valid_frames = 1e4
    # valid_frames = 1e2
    file_names = generate_raw_data(valid_frames, 'valid')

    if 1:
        print("Load test - Begin.")
        file_name = file_names[0]
        data_path = './data/' + file_name + '.h5'
        print(data_path)
        data = data_utils.load_h5_as_list(data_path)
        print('data', type(data))
        print('data[0]', type(data[0]))
        print("Load test - Success!")






