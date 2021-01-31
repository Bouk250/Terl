import time

from tests.env import mack_vec_env
from tests.custom_indicators import *
import torch

def main_func():
    venvs = mack_vec_env("TrainEnv", 1, "cuda")
    start_time = time.time() 
    obs = venvs.reset()
    obs1, obs2 = obs['market_data'], obs['portfolio_state']
    end_time = time.time() 
    print(end_time-start_time)
    print(venvs.observation_space)
    #print(ac.actor(obs), ac.critic(obs))
    reward_list = []
    print(obs1)
    for i in range(100):
        venvs.reset()
        done = False
        while not done:
            action = venvs.action_space.sample()
            obs, reward, done, _ = venvs.step([action])
            #print(action)
            #print(done)
            reward_list.append(reward)

    print(sum(reward_list))
    env = venvs._get_target_envs(0)[0].unwrapped
    print(env._portfolio._history['trade_profit'].mean())

if __name__ == '__main__':
    main_func()
