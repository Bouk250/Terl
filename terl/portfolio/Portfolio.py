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
            np.meshgrid(*([0,1,2] for _ in range(len(self._trading_price_obs))))).T.reshape(-1, len(self._trading_price_obs)),
            columns=self._trading_price_obs)
        self._num_of_action = 3**len(self._trading_price_obs)

        self._save_trade = self._config.get('save_trade')
        self._done_type = self._config.get('done_type')
        self._trade = pd.DataFrame(index=self._action_map.columns, columns=['position', 'enter_price', 'enter_datetime', 'done'])
        self._history = pd.DataFrame(columns=['symbol', 'position', 'enter_price', 'enter_datetime', 'out_price', 'out_datetime', 'trade_profit', 'trade_duration'])

    def update(self, actions:int, prices:pd.DataFrame) -> float:
        if self._done_type == "single":
            return(self._single(actions, prices))
        elif self._done_type == "multi_single":
            return self._multi_single(actions, prices)
        elif self._done_type == "multi":
            return self._multi(actions, prices)
        else:
            raise ValueError

    def _update_one_obs_price(self, action:int, key:str, prices:pd.DataFrame):
        profit = 0.0
        if action == 0:
            pass
        elif action == 1: # Buy or close sell position
            if not pd.isna(self._trade.loc[key]['position']): # Position is open
                if self._trade.loc[key]['position'] == 'Short': # Short Position is open close it
                    
                    self._trade.loc[key]['done'] = True
                    trade = self._trade.loc[key][['position', 'enter_price', 'enter_datetime']].copy()
                    
                    trade['symbol'] = trade.name

                    trade['out_price'] = prices[key]
                    trade['out_datetime'] = prices.name

                    trade['trade_profit'] = int((trade['enter_price'] - trade['out_price']) / self._pip_resolution)
                    trade['trade_duration'] = trade['out_datetime'] - trade['enter_datetime']

                    profit = trade['trade_profit']
                    if self._save_trade : self._history = self._history.append(trade, ignore_index=True)

                else: # Long Position is open do nothing
                    pass # Long Position is open do nothing
            else: # No open position
                self._trade.loc[key]['position'] = 'Long'
                self._trade.loc[key]['enter_price'] = prices[key]
                self._trade.loc[key]['enter_datetime'] = prices.name
                self._trade.loc[key]['done'] = False
                
        elif action == 2: # Sell or close buy position
            if not pd.isna(self._trade.loc[key]['position']): # Position is open
                if self._trade.loc[key]['position'] == 'Long': # Long Position is open close it
                    
                    self._trade.loc[key]['done'] = True
                    trade = self._trade.loc[key][['position', 'enter_price', 'enter_datetime']].copy()
                    
                    trade['symbol'] = trade.name
                    trade['out_price'] = prices[key]
                    trade['out_datetime'] = prices.name

                    trade['trade_profit'] = int((trade['out_price'] - trade['enter_price']) / self._pip_resolution)
                    trade['trade_duration'] = trade['out_datetime'] - trade['enter_datetime']
                    
                    profit = trade['trade_profit']
                    if self._save_trade : self._history = self._history.append(trade, ignore_index=True)
                else: # Short Position is open do nothing
                    pass # Short Position is open do nothing
            else: # No open position
                self._trade.loc[key]['position'] = 'Short'
                self._trade.loc[key]['enter_price'] = prices[key]
                self._trade.loc[key]['enter_datetime'] = prices.name
                self._trade.loc[key]['done'] = False
        return profit

    def _single(self, actions:int, prices:pd.DataFrame) -> tuple:
        profit = 0.0
        sub_actions = self._action_map.iloc[actions]
        done = False
        for index, key in  enumerate(self._trading_price_obs):
            action = sub_actions[key]
            profit += self._update_one_obs_price(action, key, prices)
            if self._trade['done'].loc[key] == True:
                self.reset()
                done = True
                break       
        return int(profit), done

    def _multi_single(self, actions:int, prices:pd.DataFrame) -> tuple:
        profit = 0.0
        sub_actions = self._action_map.iloc[actions]
        done = False
        for index, key in  enumerate(self._trading_price_obs):
            if self._trade['done'].loc[key] != True:
                action = sub_actions[key]
                profit += self._update_one_obs_price(action, key, prices)

        done = (self._trade['done'] == True).all()
        return int(profit), done

    def _multi(self, actions:int, prices:pd.DataFrame) -> tuple:
        profit = 0.0
        sub_actions = self._action_map.iloc[actions]
        for index, key in  enumerate(self._trading_price_obs):
            action = sub_actions[key]
            profit += self._update_one_obs_price(action, key, prices)
        mask = self._trade['done'] == True
        self._trade[mask] = np.NaN
        return int(profit), False

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
                legal_action = legal_action[legal_action[key] != 2]
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
                state_map = state_map[state_map[key] == 2]

        state = np.zeros(shape=(self._action_map.shape[0]))
        state[state_map.index[0]] = 1.0
        return np.copy(state)

    def clear_history(self):
        self._history = self._history.iloc[0:0]
    def reset(self):
        self._trade.loc[:] = np.NaN


        