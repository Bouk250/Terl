import time

from tests.env import mack_vec_env
import torch

def main_func():
    venvs = mack_vec_env("Env 1", 3, "cuda")
    start_time = time.time() 
    obs = venvs.reset()
    obs1, obs2 = obs['market_data'], obs['portfolio_state']
    end_time = time.time() 
    print(end_time-start_time)
    print(venvs.observation_space)
    #print(ac.actor(obs), ac.critic(obs))
    reward_list = []
    for i in range(1000):
        obs, reward, done, _ = venvs.step([0,0,0])
        obs1, obs2 = obs['market_data'], obs['portfolio_state']
        reward_list.append(reward)

    print(sum(reward_list))
    #env = venvs._get_target_envs(0)[0].unwrapped
    #print(env._portfolio._history)

if __name__ == '__main__':
    main_func()
