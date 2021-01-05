from datetime import datetime
import vaex as vx
import numpy as np
import pandas as pd
import numba as nb
from typing import Union
import gym
from gym.spaces import Discrete, Box
import terl
from terl.config import EnvConfigManager, config_checker
from terl.common.utils import random_index, load_one_file, select_from_df
from threading import Thread
from terl.portfolio import Portfolio

class TradingEnv(gym.Env):

    def __init__(self, config:Union[dict,str]):

        if type(config) is str:
            config = EnvConfigManager().get_config(config)
        config_checker(config)
        self._config = config
        self._db = dict()
        self._current_prices = None
        self._current_dt_index = 0
        self._symboles = self._config.get('symbols')
        self._timesframes = self._config.get('timeframes')
        self._data_path = self._config.get('data_path')
        self._data_loader = self._config.get('data_loader')
        self._obs_var = self._config.get('obs_var')
        self._num_of_file = len(self._symboles)*len(self._timesframes)
        self._indicators = self._config.get('indicators')
        self._num_of_history = self._config.get('num_of_history')
        start_dt = self._config.get('start_dt')
        end_dt = self._config.get('end_dt')
        self._pipeline = self._config.get('obs_pipeline')
        self._portfolio = Portfolio(self._config.get('portfolio'))
        self._trading_price_obs = self._portfolio._trading_price_obs

        self.__load_db()

        if type(start_dt) is int:
            self._min_index = start_dt
        elif type(start_dt) is datetime:
            self._min_index = np.where(self._dt_index_map.index >= datetime(2000,1,1,00,00))[0][0]
        else:
            raise ValueError()
        
        if self._min_index < self._num_of_history:
            self._min_index = self._config.get('num_of_history')

        if type(end_dt) is int:
            if end_dt == -1:
                self._max_index = self._dt_index_map.shape[0]
            else:
                self._max_index = end_dt
        elif type(end_dt) is datetime:
            self._max_index = np.where(self._dt_index_map.index >= datetime(2000,1,1,00,00))[0][0]
        else:
            raise ValueError()

        self.action_space = Discrete(self._portfolio._num_of_action)
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=self.reset().shape, dtype=np.float32)


    def get_config(self) -> dict:
        return self._config.copy()

    def set_config(self, new_config:dict):
        self._config = new_config

    def __load_db(self):
        symboles = self._symboles
        timesframes = self._timesframes
        data_path = self._data_path
        data_loader = self._data_loader
        obs_var = self._obs_var
        num_of_file = len(symboles)*len(timesframes)
        indicators = self._indicators

        index_list = [None] * num_of_file
        df_result = [None] * num_of_file
        thread_list = [None] * num_of_file
        i:int = 0
        for s in symboles:
            for t in timesframes:
                
                thread_list[i] = Thread(target=load_one_file,
                kwargs={
                    's':s, 
                    't':t, 
                    'data_loader':data_loader, 
                    'data_path':data_path, 
                    'obs_var':obs_var, 
                    'indicators':indicators, 
                    'df_result':df_result, 
                    'index_list':index_list, 
                    'i' : i
                    })
                thread_list[i].start()

                """
                load_one_file(**{
                    's':s, 
                    't':t, 
                    'data_loader':data_loader, 
                    'data_path':data_path, 
                    'obs_var':obs_var, 
                    'indicators':indicators, 
                    'df_result':df_result, 
                    'index_list':index_list, 
                    'i' : i
                    })

                """
                
                i += 1

        _ = [thread.join() for thread in thread_list]
        index_list = [i for i in index_list if i is not None]
        df_result = [df for df in df_result if df is not None]

        self._db.update(df_result)
        self._dt_index_map = pd.concat(index_list, axis=1).fillna(method='ffill').dropna().astype('int32')

    def __repr__(self) -> str:
        return str(self._config)
    
    def step(self, action):
        self._current_dt_index += 1
        self._portfolio.update(action, self._current_prices)
        obs, self._current_prices = self.__generate_obs()
        return obs


    def reset(self):
        self._current_dt_index = random_index(self._min_index, self._max_index)
        obs, self._current_prices = self.__generate_obs() 
        return obs


    def __generate_obs(self):
        obs_var = self._obs_var
        num_of_history = self._num_of_history
        df_type_vx = self._data_loader in ['vx', 'vaex']
        pipeline = self._pipeline
        db = self._db
        db_keys = db.keys()
        db_len = len(db_keys)
        indexs = self._dt_index_map.iloc[self._current_dt_index]

        obs_blocks = [None] * db_len

        for i, df_key in enumerate(db_keys):
            start_index = indexs[df_key] - num_of_history
            end_index = indexs[df_key]
            df = db.get(df_key)

            select_from_df(df,start_index,end_index,df_type_vx,obs_blocks,i)

        obs = pd.concat(obs_blocks, axis=1)[obs_var]
        prices = obs[self._trading_price_obs].iloc[-1]
        obs = obs.to_numpy(dtype=np.float32)
        if not pipeline is None:
            obs = pipeline.fit_transform(obs)
            
        return obs, prices

def make_env(config_name:str, config_path:str = None) -> TradingEnv:
    if config_path is None:
        conf = EnvConfigManager().get_config(config_name)
    else:
        conf = EnvConfigManager(config_path).get_config(config_name)
    config_checker(conf)
    return TradingEnv(conf)