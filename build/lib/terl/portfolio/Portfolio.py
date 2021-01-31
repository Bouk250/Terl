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
        self._pip_resolution = self._config.get('pip_resolution')
        self._action_map = pd.DataFrame(np.array(
            np.meshgrid(*([0,1,-1] for _ in range(len(self._trading_price_obs))))).T.reshape(-1, len(self._trading_price_obs)),
            columns=self._trading_price_obs)
        self._num_of_action = 3**len(self._trading_price_obs)

        self._save_trade = self._config.get('save_trade')
        self._trade = pd.DataFrame(index=self._action_map.columns, columns=['position', 'enter_price', 'enter_datetime'])
        self._history = pd.DataFrame(columns=['symbol', 'position', 'enter_price', 'enter_datetime', 'out_price', 'out_datetime', 'trade_profit', 'trade_duration'])

    def update(self, actions:int, prices:pd.DataFrame) -> float:
        profit = 0.0
        sub_actions = self._action_map.iloc[actions]

        for key in self._trading_price_obs:
            action = sub_actions[key]
            
            if action == 0:
                pass
            elif action == 1: # Buy or close sell position
                if not pd.isna(self._trade.loc[key]['position']): # Position is open
                    if self._trade.loc[key]['position'] == 'Short': # Short Position is open close it
                        trade = self._trade.loc[key].copy()
                        self._trade.loc[key].values[:] = np.NaN
                        trade['symbol'] = trade.name

                        trade['out_price'] = prices[key]
                        trade['out_datetime'] = prices.name

                        trade['trade_profit'] = int((trade['enter_price'] - trade['out_price']) / self._pip_resolution)
                        trade['trade_duration'] = trade['out_datetime'] - trade['enter_datetime']

                        profit += trade['trade_profit']
                        if self._save_trade : self._history = self._history.append(trade, ignore_index=True)

                    else: # Long Position is open do nothing
                        pass # Long Position is open do nothing
                else: # No open position
                    self._trade.loc[key]['position'] = 'Long'
                    self._trade.loc[key]['enter_price'] = prices[key]
                    self._trade.loc[key]['enter_datetime'] = prices.name
                    
            elif action == -1: # Sell or close buy position
                if not pd.isna(self._trade.loc[key]['position']): # Position is open
                    if self._trade.loc[key]['position'] == 'Long': # Long Position is open close it
                        trade = self._trade.loc[key].copy()
                        self._trade.loc[key].values[:] = np.NaN
                        trade['symbol'] = trade.name

                        trade['out_price'] = prices[key]
                        trade['out_datetime'] = prices.name

                        trade['trade_profit'] = int((trade['out_price'] - trade['enter_price']) / self._pip_resolution)
                        trade['trade_duration'] = trade['out_datetime'] - trade['enter_datetime']
                        
                        profit += trade['trade_profit']
                        if self._save_trade : self._history = self._history.append(trade, ignore_index=True)
                    else: # Short Position is open do nothing
                        pass # Short Position is open do nothing
                else: # No open position
                    self._trade.loc[key]['position'] = 'Short'
                    self._trade.loc[key]['enter_price'] = prices[key]
                    self._trade.loc[key]['enter_datetime'] = prices.name
        return int(profit)

    @property
    def legal_action(self) -> np.ndarray:
        legal_action = self._action_map.copy()
        for key in self._trading_price_obs:
            position = self._trade.loc[key]['position']
            if pd.isna(position):
                # Legal action is Buy, Sell, Nothing
                pass
            elif position == 'Long':
                # Legal action is Sell/Close or Nothing
                legal_action = legal_action[legal_action[key] != 1]
            elif position == 'Short':
                # Legal action is Buy/Close or Nothing
                legal_action = legal_action[legal_action[key] != -1]
        return legal_action.index.to_numpy()

    @property
    def state(self) -> np.ndarray:
        state_map = self._action_map.copy()
        for key in self._trading_price_obs:
            position = self._trade.loc[key]['position']
            if pd.isna(position):
                state_map = state_map[state_map[key] == 0]
                pass
            elif position == 'Long':
                # Legal action is Sell/Close or Nothing
                state_map = state_map[state_map[key] == 1]
            elif position == 'Short':
                # Legal action is Buy/Close or Nothing
                state_map = state_map[state_map[key] == -1]

        state = np.zeros(shape=(self._action_map.shape[0]))
        state[state_map.index[0]] = 1.0
        return state

    def reset(self):
        self._trade.loc[:] = np.NaN


        