from datetime import datetime
from threading import Thread
from typing import Union

import gym
import numpy as np
import pandas as pd
from gym.spaces import Box, Dict, Discrete, MultiBinary
from terl.common import load_one_file, random_index, select_from_df
from terl.config import EnvConfigManager, config_checker
from terl.portfolio import Portfolio
from terl.db import DBManager

class TradingEnv(gym.Env):

    def __init__(self, config: Union[dict, str], db_manager=None):

        if type(config) is str:
            config = EnvConfigManager().get_config(config)
        config_checker(config)
        self._config = config
        self._current_prices = None
        self._current_dt_index = 0
        if db_manager is None:
            self._db_manager = DBManager(self._config.get('db'))
        else:
            self._db_manager = db_manager

        #self._symboles = self._config.get('symbols')
        #self._timesframes = self._config.get('timeframes')
        #self._data_path = self._config.get('data_path')
        #self._data_loader = self._config.get('data_loader')
        #self._obs_var = self._config.get('obs_var')
        #self._indicators = self._config.get('indicators')
        #self._num_of_history = self._config.get('num_of_history')
        #start_dt = self._config.get('start_dt')
        #end_dt = self._config.get('end_dt')

        self._portfolio = Portfolio(self._config.get('portfolio'))
        self._trading_price_obs = self._portfolio._trading_price_obs



        self.action_space = Discrete(self._portfolio._num_of_action)
        self.observation_space = Dict({
            'market_data': Box(low=-np.inf, high=np.inf, shape=self.reset()['market_data'].shape, dtype=np.float32),
            'portfolio_state': MultiBinary(self._portfolio.state.shape)
        })

    def get_config(self) -> dict:
        return self._config.copy()

    def set_config(self, new_config: dict):
        self._config = new_config

    def __repr__(self) -> str:
        return str(self._config)

    def step(self, action) -> tuple:
        obs = None
        reward = 0.0
        done = False
        info = {}

        self._current_dt_index += 1
        reward = self._portfolio.update(action, self._current_prices)

        done = self._current_dt_index >= self._db_manager._max_index

        if done:
            return obs, reward, done, info

        info.update(
            {'current_dt': self._db_manager.get_datetime(self._current_dt_index)})

        obs, self._current_prices = self._db_manager.generate_obs(self._current_dt_index)

        obs = {
            'market_data': obs,
            'portfolio_state': self._portfolio.state
        }

        return obs, reward, done, info

    def reset(self) -> dict:
        self._current_dt_index = random_index(self._db_manager._min_index, self._db_manager._max_index)
        obs, self._current_prices = self._db_manager.generate_obs(self._current_dt_index)
        self._portfolio.reset()
        obs = {
            'market_data': obs,
            'portfolio_state': self._portfolio.state
        }
        return obs

def make_env(config_name: str, config_path: str = None) -> TradingEnv:
    if config_path is None:
        conf = EnvConfigManager().get_config(config_name)
    else:
        conf = EnvConfigManager(config_path).get_config(config_name)
    config_checker(conf)
    return TradingEnv(conf)
