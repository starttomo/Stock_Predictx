import pandas as pd
import numpy as np
import math

def nan_safe(val, fallback=0.0):
    if val is None or (isinstance(val, float) and (math.isnan(val) or math.isinf(val))):
        return fallback
    return float(val)

def generate_strategy(df, future):
    if len(df) == 0 or not future or 'predictions' not in future or len(future['predictions']) == 0:
        return {
            'current_analysis': "数据不足，无法分析。",
            'future_strategy': "无法生成未来策略。",
            'risk': "无法评估风险。",
            'expected_return': 0.0
        }

    # --- 当前分析
    last = df.iloc[-1]
    current_signal = ""
    if last['rsi'] < 30:
        current_signal = "超卖，潜在买入机会"
    elif last['rsi'] > 70:
        current_signal = "超买，潜在卖出机会"
    else:
        current_signal = "中性，观察"

    if last['macd'] > last['macd_signal']:
        current_signal += "；MACD金叉，趋势向上"
    else:
        current_signal += "；MACD死叉，趋势向下"

    # --- 未来分析
    future_predictions = list(zip(future['dates'], future['predictions']))
    future_df = pd.DataFrame(future_predictions, columns=['date', 'close'])
    future_df['date'] = pd.to_datetime(future_df['date'])
    extended_df = pd.concat([df.tail(20), future_df])
    close_ext = extended_df['close']

    # 未来预期涨幅
    current_close = float(last['close'])
    predicted_closes = np.array([nan_safe(x, current_close) for x in future_df['close'].values])
    if current_close:  # 防止0除
        expected_return = float(np.mean((predicted_closes - current_close) / current_close) * 100)
    else:
        expected_return = 0.0
    # 再安全一次
    expected_return = nan_safe(expected_return, 0.0)

    # 未来信号
    delta_ext = close_ext.diff()
    gain_ext = delta_ext.where(delta_ext > 0, 0).rolling(14).mean()
    loss_ext = -delta_ext.where(delta_ext < 0, 0).rolling(14).mean()
    rs_ext = gain_ext / loss_ext
    extended_df['rsi_future'] = 100 - (100 / (1 + rs_ext))
    last_future_rsi = nan_safe(extended_df['rsi_future'].iloc[-1], 50.0)

    if last_future_rsi < 30 and expected_return > 3:
        future_signal = f"预测未来RSI超卖，预期涨幅{expected_return:.2f}%，建议买入"
    elif last_future_rsi > 70 or expected_return < -3:
        future_signal = f"预测未来RSI超买，预期跌幅{-expected_return:.2f}%，建议卖出"
    else:
        future_signal = f"预测中性，预期回报{expected_return:.2f}%，建议持有"

    # 风险评估
    volatility = nan_safe(df['close'].pct_change().std() * 100 if len(df) > 1 else 0.0, 0.0)
    risk_assess = f"风险水平：波动率{volatility:.2f}%；建议设置止损于当前价-5%"

    return {
        'current_analysis': current_signal,
        'future_strategy': future_signal,
        'risk': risk_assess,
        'expected_return': expected_return
    }