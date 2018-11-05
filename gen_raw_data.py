import numpy as np
import multiprocessing as mp
import gym
import gym.spaces #TODO: remove this (only used to suppress warning

import gym_utils
import data_utils

def generate_action(env):
    a = env.action_space.sample()
    return a

def gen_data(gen_args, save=True, render=False):
    """ Format: (obs, action, reward, done)
    """
    thread_num, max_steps = gen_args

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

    env.close()
    return obs_data

def generate_raw_data(total_frames, postfix=''):
    total_frames = int(total_frames)

    num_threads = mp.cpu_count()-1 # TODO: This causes memory issues!
    num_threads = 1
    frames_per_thread = int(total_frames/num_threads) + 1
    gen_args = [(i, frames_per_thread) for i in range(num_threads)]

    print("Generating", postfix, "data for env CarRacing-v0")
    print("total_frames: ", total_frames)
    print("Threads:", num_threads, '(',frames_per_thread,'per thread)')
    print('...')

    with mp.Pool(mp.cpu_count()) as p:
        data = p.map(gen_data, gen_args)

    data_as_array = np.concatenate(data, 0)[:total_frames]

    print('Generated dataset with ', data_as_array.shape[0], "observations.")
    print("Format: (obs, action, reward, done)")
    file_name = 'Breakout_raw_'+str(data_as_array.shape[0])+'_'+postfix
    data_utils.save_np_array_as_h5(file_name, data_as_array)
    print()

    return file_name

if __name__ == '__main__':
    # TODO: This causes memory issues!
    train_frames = 1e4
    # train_frames = 10
    file_name = generate_raw_data(train_frames, 'train')

    # valid_frames = 1e3
    # file_name = generate_raw_data(valid_frames, 'valid')

    if 1:
        print("Load test - Begin.")
        data_path = './data/' + file_name + '.h5'
        print(data_path)
        data = data_utils.lad_h5_as_np_array(data_path)
        print("Load test - Success!")






