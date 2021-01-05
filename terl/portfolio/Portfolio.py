import pandas as pd
import numpy as np

"""
- symbol
- position
- enter price
- enter datetime
- out price
- out datetime
- trade profit
- trade duration
"""

class Portfolio:
    def __init__(self, config:dict):
        self._config = config

        self._trading_price_obs = self._config.get('trading_price_obs')
        self._action_map = pd.DataFrame(np.array(
            np.meshgrid(*([0,1,2] for _ in range(len(self._trading_price_obs))))).T.reshape(-1, len(self._trading_price_obs)),
            columns=self._trading_price_obs)
        self._num_of_action = 3**len(self._trading_price_obs)

        self._trade = pd.DataFrame(index=self._action_map.columns, columns=['position', 'enter_price', 'enter_datetime', 'out_price', 'trade_profit', 'trade_duration'])
        self._history = pd.DataFrame(columns=['symbol', 'position', 'enter_price', 'enter_datetime', 'out_price', 'trade_profit', 'trade_duration'])

    def update(self, actions, prices):
        sub_actions = self._action_map.iloc[actions]

        for key in self._trading_price_obs:
            action = sub_actions[key]
            pass

    def legal_action(self):
        for key in self._trading_price_obs:
            position = self._trade[key]['position']
            if position is pd.NaT:
                # Legal action is Buy, Sell, Nothing
                pass
            elif position == 'Long':
                # Legal action is Sell/Close or Nothing
                pass
            elif position == 'Short':
                # Legal action is Buy/Close or Nothing
                pass

    def reset(self):
        pass


        