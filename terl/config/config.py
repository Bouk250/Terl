_default_config = {
    'symbols':[],
    'timeframes':[]
}

def GetDefaultConf():
    return _default_config.copy()

def SetDefaultConf(config:dict):
    _default_config = config