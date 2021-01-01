import os
import vaex as vx
import numpy as np
import pandas as pd
import numba as nb
import gym
from gym.spaces import MultiDiscrete
import terl
from terl.config import EnvConfigManager, config_checker


class TradingEnv(gym.Env):
    def __init__(self, config:dict):
        self._config = config
        self._db = dict()
        self._dt_index_map = None

        self.__load_db()

    def get_config(self) -> dict:
        return self._config.copy()

    def set_config(self, new_config:dict):
        self._config = new_config

    def __load_db(self):
        symboles = self._config.get('symbols')
        timesframes = self._config.get('timeframes')
        data_path = self._config.get('data_path')
        data_loader = self._config.get('data_loader')

        index_list = []



        for s in symboles:
            for t in timesframes:
                df_id = f"{s}_{t}"
                if data_loader in ["pd","pandas"]:
                    file_path = os.path.join(data_path,s,f"{df_id}.h5")
                    pd_df = pd.read_hdf(file_path)

                elif data_loader in ["vx", "vaex"]:
                    file_path = os.path.join(data_path,s,f"{df_id}.hdf5")
                    vx_df = vx.open(file_path)

                    index = pd.DataFrame(index=vx_df['time'].to_numpy(), data=np.arange(0,vx_df.shape[0]), columns=[df_id])
                    index_list.append(index)

                    vx_df = vx_df.drop('time')
                    self._db.update([(df_id,vx_df)])

        self._dt_index_map = pd.concat(index_list, axis=1).fillna(method='ffill')



def make_env(config_name:str, config_path:str = None) -> TradingEnv:
    if config_path is None:
        conf = EnvConfigManager().get_config(config_name)
    else:
        conf = EnvConfigManager(config_path).get_config(config_name)
    config_checker(conf)
    return TradingEnv(conf)