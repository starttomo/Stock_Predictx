# utils/indicators.py
import pandas as pd
import numpy as np


def add_indicators(df):
    close = df['close']

    # MA
    df['ma7'] = close.rolling(7).mean()
    df['ma30'] = close.rolling(30).mean()
    df['ma60'] = close.rolling(60).mean()

    # RSI
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.rolling(14).mean()
    ma_down = down.rolling(14).mean()
    rs = ma_up / ma_down
    df['rsi'] = 100 - (100 / (1 + rs))

    # BOLL
    df['mid'] = close.rolling(20).mean()
    std = close.rolling(20).std()
    df['upper'] = df['mid'] + 2 * std
    df['lower'] = df['mid'] - 2 * std

    return df