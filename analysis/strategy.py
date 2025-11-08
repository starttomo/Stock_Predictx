# analysis/strategy.py
import pandas as pd
import numpy as np

# 经济学依据：基于弱形式有效市场假设（Fama, 1970）和行为金融学（Kahneman & Tversky, 1979），
# 使用TA指标捕捉历史价格模式。整合LSTM预测符合预期理论（Muth, 1961），参考论文：Agrawal et al. (2021)
# "Stock Price Prediction using Technical Indicators: A Predictive Model using Optimal Deep Learning"，
# 证明TA+数值预测提升策略准确率20-30%。

def generate_strategy(df, future_predictions):
    """
    生成交易策略：结合历史指标和模型预测。
    - 历史部分：用RSI、MACD、BOLL等定性分析当前信号。
    - 预测部分：用未来yhat计算未来指标，量化预期涨幅/风险。
    - 输出：字典，包括当前建议、未来预期、风险评估。
    """
    # 历史分析（定性）
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

    # 预测整合（量化）：用未来yhat扩展df，计算未来指标
    future_df = pd.DataFrame(future_predictions, columns=['date', 'close'])
    future_df['date'] = pd.to_datetime(future_df['date'])
    extended_df = pd.concat([df.tail(20), future_df])  # 用最后20天历史+预测，计算滚动指标

    # 计算未来指标（简化自回归：重新计算RSI、MA等）
    close_ext = extended_df['close']
    # 未来MA
    extended_df['ma5_future'] = close_ext.rolling(5).mean()
    # 未来RSI
    delta_ext = close_ext.diff()
    gain_ext = delta_ext.where(delta_ext > 0, 0).rolling(14).mean()
    loss_ext = -delta_ext.where(delta_ext < 0, 0).rolling(14).mean()
    rs_ext = gain_ext / loss_ext
    extended_df['rsi_future'] = 100 - (100 / (1 + rs_ext))
    # 未来预期涨幅
    current_close = last['close']
    predicted_closes = future_df['close'].values
    expected_return = np.mean((predicted_closes - current_close) / current_close) * 100  # 平均预期回报率

    # 未来信号
    future_signal = ""
    if extended_df['rsi_future'].iloc[-1] < 30 and expected_return > 3:
        future_signal = f"预测未来RSI超卖，预期涨幅{expected_return:.2f}%，建议买入"
    elif extended_df['rsi_future'].iloc[-1] > 70 or expected_return < -3:
        future_signal = f"预测未来RSI超买，预期跌幅{-expected_return:.2f}%，建议卖出"
    else:
        future_signal = f"预测中性，预期回报{expected_return:.2f}%，建议持有"

    # 风险评估（用置信区间，假设future_predictions有lower/upper，或简化用波动率）
    volatility = df['close'].pct_change().std() * 100  # 历史波动率
    risk_assess = f"风险水平：波动率{volatility:.2f}%；建议设置止损于当前价-5%"

    return {
        'current_analysis': current_signal,
        'future_strategy': future_signal,
        'risk': risk_assess,
        'expected_return': expected_return
    }