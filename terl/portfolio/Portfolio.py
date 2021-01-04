import pandas as pd
import numpy as np

class Portfolio:
    def __init__(self, config:dict):
        self._config = config

        self._trading_price_obs = self._config.get('trading_price_obs')
        self._action_map = pd.DataFrame(np.array(
            np.meshgrid(*([0,1,2] for _ in range(len(self._trading_price_obs))))).T.reshape(-1, len(self._trading_price_obs)),
            columns=self._trading_price_obs)
        self._num_of_action = 3**len(self._trading_price_obs)

        