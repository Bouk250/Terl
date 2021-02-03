from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from gym import ObservationWrapper, RewardWrapper
from gym.spaces import Box, Dict
import numpy as np
from pyts.image import GramianAngularField

class ForexWrapper(ObservationWrapper):
    def __init__(self, env,axis=None, min_val=0, max_val=1):
        super(ForexWrapper, self).__init__(env)
        
        ct = make_column_transformer((FunctionTransformer(self.minmax_scale, kw_args={'axis':axis, 'feature_range':(min_val,max_val)}),[0,1,2,3]),
                                     (FunctionTransformer(self.minmax_scale, kw_args={'axis':axis, 'feature_range':(min_val,max_val)}),[4]),
                                     remainder='passthrough')

        gadf = GramianAngularField(image_size=60, method='difference')
        self.model = make_pipeline(ct, FunctionTransformer(self.T), gadf)

        self.observation_space = Box(low=min_val, high=max_val, 
                        shape=self.model.fit_transform(np.random.random(size=self.observation_space['market_data'].shape)).shape, 
                        dtype=np.float32)
        """
        Dict({
            'market_data':Box(low=min_val, high=max_val, 
                        shape=self.model.fit_transform(np.random.random(size=self.observation_space['market_data'].shape)).shape, 
                        dtype=np.float32),
            'portfolio_state':self.observation_space['portfolio_state']
        }) 
        """
    
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
        observation = self.model.fit_transform(observation['market_data'])
        return observation


class RewardWrapper(RewardWrapper):
    def __init__(self, env, reward_penality=0, min_reward=-np.inf, max_reward=np.inf):
        super(RewardWrapper,self).__init__(env)

        self.reward_penality = reward_penality
        self.min_reward = min_reward
        self.max_reward = max_reward

    def reward(self, reward):
        reward += self.reward_penality
        reward = max(reward, self.min_reward)
        reward = min(reward, self.max_reward)
        return reward