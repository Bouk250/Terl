from stable_baselines3.common.vec_env import VecEnvWrapper
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from gym import ObservationWrapper
from gym.spaces import Box, Dict, MultiBinary
import numpy as np
import torch

class ForexWrapper(ObservationWrapper):
    def __init__(self, env,axis=None, min_val=0, max_val=1):
        super(ForexWrapper, self).__init__(env)
        
        ct = make_column_transformer((FunctionTransformer(self.minmax_scale, kw_args={'axis':axis, 'feature_range':(min_val,max_val)}),[0,1,2,3]),
                                     (FunctionTransformer(self.minmax_scale, kw_args={'axis':axis, 'feature_range':(min_val,max_val)}),[4]),
                                     remainder='passthrough')
        
        self.model = make_pipeline(ct, FunctionTransformer(self.T))

        self.observation_space = Dict({
            'market_data':Box(low=min_val, high=max_val, 
                        shape=self.model.fit_transform(np.random.random(size=self.observation_space['market_data'].shape)).shape, 
                        dtype=np.float32),
            'portfolio_state':self.observation_space['portfolio_state']
        }) 
    
    @staticmethod
    def minmax_scale(X, axis=0, feature_range=(0,1)):
        X_out = np.zeros_like(X)
        X_min = np.min(X, axis=axis)
        X_max = np.max(X, axis=axis)

        X_out = (X - X_min) / (X_max - X_min)
        X_out = X_out * (feature_range[1]-feature_range[0]) + feature_range[0]

        return X_out
    
    @staticmethod
    def T(X:np.ndarray):
        return X.T
        
    def observation(self, observation):
        observation['market_data'] = self.model.fit_transform(observation['market_data'])
        return observation

class VecPyTorch(VecEnvWrapper):
    def __init__(self, venv, device):
        """Return only every `skip`-th frame"""
        super(VecPyTorch, self).__init__(venv)
        self.device = device

    def reset(self):
        obs = self.venv.reset()
        for sub_obs in obs:
            obs[sub_obs] = torch.from_numpy(obs[sub_obs]).float().to(self.device)
        return obs

    def step_async(self, actions):
        actions = actions.cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        for sub_obs in obs:
            obs[sub_obs] = torch.from_numpy(obs[sub_obs]).float().to(self.device)
        reward = torch.from_numpy(reward).unsqueeze(dim=1).float()
        return obs, reward, done, info