from threading import Thread
import numba
import os
import numpy as np
import pandas as pd
import vaex as vx

@numba.njit(fastmath=True)
def random_index(min_index:int, max_index:int) -> int:
    return np.random.randint(min_index, max_index,size=1)[0]

def load_one_file(s:str,t:int,data_loader:str, data_path:str, obs_var:list, indicators:dict, df_result:list,index_list:list, i:int):
    df_id = f"{s}_{t}"
    df = None
    index = None

    if data_loader in ['pd','pandas']:
        file_path = os.path.join(data_path,s,f"{df_id}.h5")
        df = pd.read_hdf(file_path)
        index = pd.DataFrame(index=df['time'].to_numpy(), data=np.arange(0,df.shape[0], dtype=np.int32), columns=[df_id])
        df = df.drop('time', axis=1)
        
        if indicators is not None:
            for key in indicators:
                indicator = indicators.get(key, dict)
                symbols = indicator.get('symbols')
                timeframes = indicator.get('timeframes')
                if s in symbols and t in timeframes:
                    kwargs=dict()
                    indicator_func = indicator.get('indicator_func')
                    series = indicator.get('series')
                    params = indicator.get('params')
                    for series_key in series:
                        kwargs.update([(series_key, df[series.get(series_key)])])
                    kwargs.update(params)
                    df[key] = indicator_func(**kwargs).to_numpy()
        
        df = df.add_prefix(f"{df_id}_")

    elif data_loader in ['vx', 'vaex']:
        file_path = os.path.join(data_path,s,f"{df_id}.hdf5")
        df = vx.open(file_path)
        index = pd.DataFrame(index=df['time'].to_numpy(), data=np.arange(0,df.shape[0], dtype=np.int32), columns=[df_id])
        df = df.drop('time')

        if indicators is not None:
            for key in indicators:
                indicator = indicators.get(key, dict)
                symbols = indicator.get('symbols')
                timeframes = indicator.get('timeframes')
                if s in symbols and t in timeframes:
                    kwargs=dict()
                    indicator_func = indicator.get('indicator_func')
                    series = indicator.get('series')
                    params = indicator.get('params')
                    for series_key in series:
                        kwargs.update([(series_key, df[series.get(series_key)].to_pandas_series())])
                    kwargs.update(params)
                    df[key] = indicator_func(**kwargs).to_numpy()

        for col in list(df.columns):
            new_col = f"{df_id}_{col}"
            df.rename(col, new_col)

    intersection = list(set(df.columns) & set(obs_var)) 
    obs_df = df[intersection]
    if obs_df.shape[0]>0:
        index_list[i] = index
        df_result[i] = (df_id,obs_df)

def select_from_df(df, start_index, end_index, is_vaex_df, result, index):
    if is_vaex_df:
        result[index] = df[start_index:end_index].to_pandas_df()
    else:
        result[index] = df.iloc[start_index:end_index].reset_index()
