from ta.momentum import *
from ta.others import *
from ta.volume import *
from ta.trend import *
from ta.utils import *
from ta.volatility import *

import pandas as pd


class DSS:
    def __init__(self, close: pd.Series, high: pd.Series, low: pd.Series, stoch_period: int = 13, ema_period: int = 8):
        self._close = close
        self._high = high
        self._low = low
        self._stoch_period = stoch_period
        self._ema_period = ema_period
        self._run()

    def _run(self):
        x = stoch(close=self._close, high=self._high, low=self._low, window=self._stoch_period)
        x = ema_indicator(x, window=self._ema_period)
        x = stoch(close=x, high=x, low=x, window=self._stoch_period)
        self._DSS = ema_indicator(x, window=self._ema_period)

    def dss(self) -> pd.Series:
        return pd.Series(self._DSS, name="DSS")


def dss(close: pd.Series, high: pd.Series, low: pd.Series, stoch_period: int = 13, ema_period: int = 8) -> pd.Series:
    return DSS(close=close, high=high, low=low, stoch_period=stoch_period, ema_period=ema_period).dss()
