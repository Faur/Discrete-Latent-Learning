import gym

env = gym.make("Breakout-v0")

def reset_env(env):
    obs = env.reset()
    reward_sum = 0
    done = False
    return obs, reward_sum, done

obs, reward_sum, done = reset_env(env)
while(True):
    obs, reward, done, info = env.step(env.action_space.sample())
    reward_sum += reward
    env.render()

    if done:
        print('Episode reward', reward_sum)
        obs, reward_sum, done = reset_env(env)

