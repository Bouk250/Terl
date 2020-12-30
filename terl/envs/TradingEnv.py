import vaex as vx
import numpy as np
import pandas as pd
import numba as nb
import gym
from gym.spaces import MultiDiscrete
from terl.envs.config import GetDefaultConf

class TradingEnv(gym.Env):
    def __init__(self, config:dict = GetDefaultConf()):
        self._config = config

    

    @staticmethod
    def _read_db():
        pass