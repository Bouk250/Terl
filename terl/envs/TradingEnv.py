from datetime import datetime
import os
from sklearn.utils.validation import check_array
import vaex as vx
import numpy as np
import pandas as pd
import numba as nb
import gym
from gym.spaces import MultiDiscrete
import terl
from terl.config import EnvConfigManager, config_checker
from sklearn.pipeline import Pipeline, make_pipeline
import numba
from numba import prange
from terl.common.utils import random_index

class TradingEnv(gym.Env):
    def __init__(self, config:dict, obs_preprocessing_pipline:Pipeline = None):
        config_checker(config)
        self._config = config
        self._db = dict()

        self._dt_index_map = None
        self.__load_db()

        self._current_dt_index = 0

        start_dt = self._config.get('start_dt')
        end_dt = self._config.get('end_dt')

        if type(start_dt) is int:
            self._min_index = start_dt
        elif type(start_dt) is datetime:
            self._min_index = np.where(self._dt_index_map.index >= datetime(2000,1,1,00,00))[0][0]
        else:
            raise ValueError()
        
        if self._min_index < self._config.get('num_of_history'):
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


    def get_config(self) -> dict:
        return self._config.copy()

    def set_config(self, new_config:dict):
        self._config = new_config

    def __load_db(self):
        symboles = self._config.get('symbols')
        timesframes = self._config.get('timeframes')
        data_path = self._config.get('data_path')
        data_loader = self._config.get('data_loader')
        obs_var = self._config.get('obs_var')

        index_list = []

        for s in symboles:
            for t in timesframes:
                df_id = f"{s}_{t}"
                df = None
                index = None

                if data_loader in ['pd','pandas']:
                    file_path = os.path.join(data_path,s,f"{df_id}.h5")
                    df = pd.read_hdf(file_path)
                    index = pd.DataFrame(index=df['time'].to_numpy(), data=np.arange(0,df.shape[0], dtype=np.int32), columns=[df_id])
                    df = df.drop('time', axis=1)
                    df = df.add_prefix(f"{df_id}_")

                elif data_loader in ['vx', 'vaex']:
                    file_path = os.path.join(data_path,s,f"{df_id}.hdf5")
                    df = vx.open(file_path)
                    index = pd.DataFrame(index=df['time'].to_numpy(), data=np.arange(0,df.shape[0], dtype=np.int32), columns=[df_id])
                    df = df.drop('time')

                    for col in list(df.columns):
                        new_col = f"{df_id}_{col}"
                        df.rename(col, new_col)

                intersection = list(set(df.columns) & set(obs_var)) 
                obs_df = df[intersection]
                if obs_df.shape[1]>0:
                    index_list.append(index)
                    self._db.update([(df_id,obs_df)])

        self._dt_index_map = pd.concat(index_list, axis=1).fillna(method='ffill').dropna().astype('int32')
    
    def step(self, action):
        self._current_dt_index += 1
        obs_var = self._config.get('obs_var')
        num_of_history = self._config.get('num_of_history')
        df_type_vx = self._config.get('data_loader') in ['vx', 'vaex']
        pipeline = self._config.get('obs_pipeline')
        db = self._db
        db_keys = db.keys()
        db_len = len(db_keys)
        indexs = self._dt_index_map.iloc[self._current_dt_index]


        obs_blocks = [None] * db_len

        for i, df_key in enumerate(db_keys):
            start_index = indexs[df_key] - num_of_history
            end_index = indexs[df_key]
            df = db.get(df_key)
            self.__select_from_df(df,start_index,end_index,df_type_vx,obs_blocks,i)

        obs = pd.concat(obs_blocks, axis=1)[obs_var].to_numpy()
        if not pipeline is None:
            obs = pipeline.fit_transform(obs)
            
        return obs


    def reset(self):
        self._current_dt_index = random_index(self._min_index, self._max_index)

        obs_var = self._config.get('obs_var')
        num_of_history = self._config.get('num_of_history')
        df_type_vx = self._config.get('data_loader') in ['vx', 'vaex']
        pipeline = self._config.get('obs_pipeline')
        db = self._db
        db_keys = db.keys()
        db_len = len(db_keys)
        indexs = self._dt_index_map.iloc[self._current_dt_index]


        obs_blocks = [None] * db_len

        for i, df_key in enumerate(db_keys):
            start_index = indexs[df_key] - num_of_history
            end_index = indexs[df_key]
            df = db.get(df_key)
            self.__select_from_df(df,start_index,end_index,df_type_vx,obs_blocks,i)

        obs = pd.concat(obs_blocks, axis=1)[obs_var].to_numpy()
        if not pipeline is None:
            obs = pipeline.fit_transform(obs)
            
        return obs

    @staticmethod
    def __select_from_df(df, start_index, end_index, is_vaex_df, result, index):
        if is_vaex_df:
            result[index] = df[start_index:end_index].to_pandas_df()
        else:
            result[index] = df.iloc[start_index:end_index].reset_index()


def make_env(config_name:str, config_path:str = None) -> TradingEnv:
    if config_path is None:
        conf = EnvConfigManager().get_config(config_name)
    else:
        conf = EnvConfigManager(config_path).get_config(config_name)
    config_checker(conf)
    return TradingEnv(conf)