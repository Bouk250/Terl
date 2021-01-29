import numpy as np
import pandas as pd
from terl.common import load_one_file, random_index, select_from_df
from threading import Thread
from datetime import datetime

class DBManager:

    def __init__(self, config: dict):
        
        self._db = dict()
        self._config = config
        self._symboles = self._config.get('symbols')
        self._timesframes = self._config.get('timeframes')
        self._data_path = self._config.get('data_path')
        self._data_loader = self._config.get('data_loader')
        self._obs_var = self._config.get('obs_var')
        self._indicators = self._config.get('indicators')
        self._num_of_history = self._config.get('num_of_history')
        self._dt_index_map = None
        self._start_dt = self._config.get('start_dt')
        self._end_dt = self._config.get('end_dt')
        self.load_db()

    def load_db(self):
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

        for thread in thread_list:
            thread.join()
        index_list = [i for i in index_list if i is not None]
        df_result = [df for df in df_result if df is not None]

        self._db.update(df_result)
        self._dt_index_map = pd.concat(index_list, axis=1).fillna(method='ffill').dropna().astype('int32')

        if type(self._start_dt) is int:
            self._min_index = self._start_dt
        elif type(self._start_dt) is datetime:
            self._min_index = np.where(
                self._dt_index_map.index >= self._start_dt)[0][0]
        else:
            raise ValueError()

        if self._min_index < self._num_of_history:
            self._min_index = self._config.get('num_of_history')

        if type(self._end_dt) is int:
            if self._end_dt == -1:
                self._max_index = self._dt_index_map.shape[0]
            else:
                self._max_index = self._end_dt
        elif type(self._end_dt) is datetime:
            self._max_index = np.where(
                self._dt_index_map.index >= self._end_dt)[0][0]
        else:
            raise ValueError()
    

    def generate_obs(self, current_dt_index) -> tuple:

        obs_var = self._obs_var
        num_of_history = self._num_of_history
        df_type_vx = self._data_loader in ['vx', 'vaex']
        db = self._db
        db_keys = db.keys()
        db_len = len(db_keys)
        indexs = self._dt_index_map.iloc[current_dt_index]

        obs_blocks = [None] * db_len

        for i, df_key in enumerate(db_keys):
            start_index = indexs[df_key] - num_of_history
            end_index = indexs[df_key]
            df = db.get(df_key)

            select_from_df(df, start_index, end_index,
                           df_type_vx, obs_blocks, i)

        obs = pd.concat(obs_blocks, axis=1)[obs_var]
        prices = obs.iloc[-1]
        prices.name = indexs.name
        obs = obs.to_numpy(dtype=np.float32)

        return obs, prices

    def get_datetime(self, index):
        return self._dt_index_map.index[index]