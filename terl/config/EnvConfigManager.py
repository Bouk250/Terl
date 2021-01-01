import yaml
from yaml.parser import ParserError


class EnvConfigManager:

    __doc__ = """
    EnvConfigManager is a manager for env config you can:
      1. get empty new config
      2. get config by name
      3. save new config
      3. update config
    """

    __default_empty_config = {
        "symbols" : ["GBPUSD"],
        "timeframes" : [15],
        "data_path": "/",
        "data_loader": "veax"
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
        return self._env_config.get(config_name)

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
    if config is None:
        raise ValueError()
    empty_config_keys = get_new_config().keys()
    for key in empty_config_keys:
        if not key in config.keys():
            raise ValueError()
        param = config.get(key)
        if param is None:
            raise ValueError()
        if len(param) == 0:
            raise ValueError()