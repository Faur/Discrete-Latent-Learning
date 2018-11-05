import gym

def reset_env(env):
    obs = env.reset()
    reward_sum = 0
    done = False
    return obs, reward_sum, done

env = gym.make("Breakout-v0")
obs, reward_sum, done = reset_env(env)
steps = 0
while(True):
    obs, reward, done, info = env.step(env.action_space.sample())
    steps += 1
    reward_sum += reward
    env.render()

    if steps >= 512:
        done = True

    if done:
        print('Episode reward: {:5}'.format(reward_sum), end='. ')
        print('Num steps: {:5}'.format(steps), end='. ')
        print()
        obs, reward_sum, done = reset_env(env)
        steps = 0

