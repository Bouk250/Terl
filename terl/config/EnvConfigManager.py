import yaml
from yaml.parser import ParserError
from datetime import datetime
from ta.momentum import rsi

class EnvConfigManager:

    __doc__ = """
    EnvConfigManager is a manager for env config you can:
      1. get empty new config
      2. get config by name
      3. save new config
      3. update config
    """

    __default_empty_config = {
        "symbols" : ['GBPUSD'],
        "timeframes" : [15],
        "data_path": '/',
        "data_loader": 'vaex',
        "num_of_history" : 60,
        "trading_price_obs":['GBPUSD_15_close'],
        "start_dt" : datetime(2000,1,1,00,00),
        "end_dt": -1,
        "obs_var" : ['GBPUSD_15_close'],
        "indicators": {
            'rsi': {
                'indicator_func': rsi,
                'series': {
                    'close': 'close'
                },
                'params': {
                    'window': 12
                }
            }
        },
        "obs_pipeline":None,
    }
    
    def __init__(self, config_path:str = "config.yaml", new_config_file:bool = False):
        self._env_config = dict()
        if new_config_file:
            self.__wirte_config(config_path, self._env_config)
        else:
            self._env_config = self.__read_config(config_path)

        self._config_path = config_path 

    
    def save_config(self,config:dict, config_name:str):
        self._env_config.update([(config_name, config)])
        self.__wirte_config(self._config_path, self._env_config)


    def remove_config(self, config_name:str):
        self._env_config.pop(config_name, None)

    def get_config(self, config_name:str) -> dict:
        return self._env_config.get(config_name).copy()

    @staticmethod
    def __wirte_config(config_path:str, env_config:dict):
        with open(config_path, 'w') as stream:
            yaml.dump(env_config, stream=stream)

    @staticmethod
    def __read_config(config_path:str) -> dict:
        try:
            with open(config_path, 'r') as stream:
                return yaml.load(stream=stream)
        except FileNotFoundError:
            raise FileNotFoundError("Config file not exist")
        except ParserError:
            raise ParserError("Error when parsing the config file")

    @classmethod
    def config_checker(cls, config:dict):
        config_checker(config)
    
    @classmethod
    def get_new_config(cls) -> dict:
        return cls.__default_empty_config.copy()

def get_new_config() -> dict:
    return EnvConfigManager.get_new_config()

def config_checker(config:dict):
    return None
    symboles = config.get('symbols')
    timesframes = config.get('timeframes')
    data_path = config.get('data_path')
    data_loader = config.get('data_loader')
    obs_var = config.get('obs_var')
    obs_var = config.get('obs_var')
    num_of_history = config.get('num_of_history')
    df_type_vx = config.get('data_loader') in ['vx', 'vaex']
    pipeline = config.get('obs_pipeline')
    start_dt = config.get('start_dt')
    end_dt = config.get('end_dt')