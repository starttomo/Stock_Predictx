# utils/strategy.py
def generate_strategy(df):
    if df.empty:
        return "<p>暂无数据</p>"

    latest = df.iloc[-1]
    rsi = latest.get('rsi', 50)
    close = latest.get('close', 0)
    ma7 = latest.get('ma7', 0)
    ma30 = latest.get('ma30', 0)
    upper = latest.get('upper', 0)
    lower = latest.get('lower', 0)

    analysis = f"""
    <h4>量化分析报告</h4>
    <p><strong>当前收盘价：</strong>{close:.2f}</p>
    <p><strong>RSI(14)：</strong>{rsi:.2f} → {'超卖' if rsi < 30 else '超买' if rsi > 70 else '中性'}</p>
    <p><strong>价格 vs 布林带：</strong>{'价格触及上轨' if close >= upper else '价格触及下轨' if close <= lower else '价格在轨道内'}</p>
    <p><strong>短期均线(MA7) vs 长期均线(MA30)：</strong>{'金叉' if ma7 > ma30 else '死叉'}</p>
    <p><strong>价格 vs 短期均线：</strong>{'价格在短期均线上方' if close > ma7 else '价格在短期均线下方'}</p>
    <p><strong>价格 vs 长期均线：</strong>{'价格在长期均线上方' if close > ma30 else '价格在长期均线下方'}</p>
    """

    # 生成买卖建议
    signals = []

    if rsi < 30 and close <= lower:
        signals.append('超卖信号：考虑买入')
    if rsi > 70 and close >= upper:
        signals.append('超买信号：考虑卖出')
    if ma7 > ma30 and close > ma7:
        signals.append('多头排列：看涨')
    if ma7 < ma30 and close < ma7:
        signals.append('空头排列：看跌')

    if signals:
        analysis += '<h4>买卖建议</h4>'
        for signal in signals:
            analysis += f'<p>{signal}</p>'
    else:
        analysis += '<h4>买卖建议</h4><p>当前市场信号不明确，建议观望</p>'

    return analysis