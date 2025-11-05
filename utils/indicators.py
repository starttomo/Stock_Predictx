# utils/indicators.py
import pandas as pd
import numpy as np
from datetime import timedelta

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

def simulate_order(df, order_date, close_date, price, stop_loss, take_profit):
    """基于历史数据模拟挂单和触发"""
    order_date = pd.to_datetime(order_date)
    close_date = pd.to_datetime(close_date)
    period_df = df[(df['date'] >= order_date) & (df['date'] <= close_date)].copy()

    # 检查挂单成功
    day_data = period_df.iloc[0]
    if price and not (day_data['low'] <= price <= day_data['high']):
        return False, "挂单价格不在当日高低价之间"

    # 检查止盈止损
    sell_date = close_date
    sell_price = period_df.iloc[-1]['close']
    for i, row in period_df.iterrows():
        if (stop_loss and row['low'] <= stop_loss) or (take_profit and row['high'] >= take_profit):
            sell_date = row['date']
            sell_price = min(max((stop_loss or take_profit), row['low']), row['high'])  # 模拟触发价
            break

    # 返回格式修正：返回三个值而不是四个
    chart_data = {
        'dates': period_df['date'].dt.strftime('%Y-%m-%d').tolist(),
        'prices': period_df['close'].tolist()
    }
    
    return True, (chart_data, sell_date, sell_price)