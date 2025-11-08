# model/predict.py - 未来预测独立模块
import torch
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import joblib
import os
from data.loader import get_hs300_data
from model.lstm_model import ImprovedLSTMModel
from database.models import db, Forecast


def create_features(df, is_predict=False):
    """创建技术指标特征"""
    close = df['close']

    # 价格特征
    df['returns'] = close.pct_change()
    df['log_returns'] = np.log(close / close.shift(1))

    # 移动平均线 - 预测时用min(window, len(df))避免NaN
    for window in [5, 10, 20, 30, 60]:
        roll_win = min(window, len(df)) if is_predict else window
        df[f'ma{window}'] = close.rolling(roll_win).mean()
        df[f'ma_ratio_{window}'] = close / df[f'ma{window}']

    # 波动率
    df['volatility_5'] = df['returns'].rolling(min(5, len(df))).std()
    df['volatility_20'] = df['returns'].rolling(min(20, len(df))).std()

    # 动量指标
    df['momentum_5'] = close / close.shift(5) - 1
    df['momentum_10'] = close / close.shift(10) - 1

    # 布林带
    df['bb_middle'] = close.rolling(min(20, len(df))).mean()
    bb_std = close.rolling(min(20, len(df))).std()
    df['bb_upper'] = df['bb_middle'] + 2 * bb_std
    df['bb_lower'] = df['bb_middle'] - 2 * bb_std
    df['bb_position'] = (close - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

    # RSI
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(min(14, len(df))).mean()
    loss = -delta.where(delta < 0, 0).rolling(min(14, len(df))).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # MACD
    exp1 = close.ewm(span=12).mean()
    exp2 = close.ewm(span=26).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=9).mean()

    # 成交量特征
    df['volume_ma'] = df['volume'].rolling(min(20, len(df))).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma']

    # 价格位置特征
    if not is_predict:
        df['high_52'] = close.rolling(252).max()
        df['low_52'] = close.rolling(252).min()
        df['price_position'] = (close - df['low_52']) / (df['high_52'] - df['low_52'])
    else:
        df['high_52'] = close.max()
        df['low_52'] = close.min()
        df['price_position'] = (close - df['low_52']) / (df['high_52'] - df['low_52'])

    # 强制填充所有NaN
    return df.ffill().bfill().fillna(0)


def prepare_sequences_multivariate(df, seq_len, target_col='close'):
    """准备多变量序列数据"""
    feature_cols = [col for col in df.columns if col != 'date']
    features = df[feature_cols].values
    targets = df[target_col].values

    X, y = [], []
    for i in range(len(features) - seq_len):
        X.append(features[i:i + seq_len])
        y.append(targets[i + seq_len])

    return np.array(X), np.array(y)


def predict_future():
    """预测未来5个交易日的价格"""
    print("开始未来预测...")

    # 检查模型和scaler是否存在
    if not os.path.exists('./model/model/best_model.pth'):
        print("错误：未找到训练好的模型文件 './model/model/best_model.pth'")
        return None

    if not os.path.exists('./model/model/scaler_X.pkl') or not os.path.exists('./model/model/scaler_y.pkl'):
        print("错误：未找到scaler文件")
        return None

    # 获取最新数据
    df = get_hs300_data()
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['date'] > '2020-01-01'].reset_index(drop=True)
    print(f"使用数据行数: {len(df)}")

    # 创建特征
    df = create_features(df)

    # 加载scaler
    try:
        scaler_X = joblib.load('./model/model/scaler_X.pkl')
        scaler_y = joblib.load('./model/model/scaler_y.pkl')
    except Exception as e:
        print(f"加载scaler失败: {e}")
        return None

    # 准备序列
    feature_cols = [col for col in df.columns if col != 'date']
    data_scaled = scaler_X.transform(df[feature_cols])
    scaled_df = pd.DataFrame(data_scaled, columns=feature_cols, index=df.index)
    scaled_df['date'] = df['date']

    seq_len = 60
    X, y = prepare_sequences_multivariate(scaled_df, seq_len)

    if len(X) == 0:
        print("错误：没有足够的数据生成序列")
        return None

    # 加载模型
    input_size = X.shape[2]
    model = ImprovedLSTMModel(input_size=input_size, hidden_size=100, num_layers=3, output_size=1)

    try:
        model.load_state_dict(torch.load('./model/model/best_model.pth', map_location='cpu'))
    except Exception as e:
        print(f"加载模型失败: {e}")
        return None

    # 预测未来
    model.eval()
    last_sequence = X[-1:].copy()
    last_df = df.tail(252).copy()

    future_dates = []
    future_preds = []
    current_date = df['date'].iloc[-1]

    with torch.no_grad():
        temp_sequence = last_sequence.copy()

        for i in range(5):
            # 计算下一个交易日（跳过周末）
            next_date = current_date + timedelta(days=1)
            while next_date.weekday() >= 5:
                next_date += timedelta(days=1)

            # 预测
            pred_scaled = model(torch.FloatTensor(temp_sequence)).numpy()[0, 0]
            pred_price = scaler_y.inverse_transform([[pred_scaled]])[0, 0]

            future_preds.append(pred_price)
            future_dates.append(next_date)

            # 更新数据用于下一次预测
            new_row = last_df.iloc[-1].copy()
            new_row['date'] = next_date
            new_row['close'] = pred_price
            new_row['open'] = pred_price
            new_row['high'] = pred_price * 1.01
            new_row['low'] = pred_price * 0.99
            new_row['volume'] = last_df['volume'].mean()

            last_df = pd.concat([last_df, pd.DataFrame([new_row])], ignore_index=True)
            last_df = create_features(last_df, is_predict=True).tail(seq_len)

            # 更新序列
            try:
                new_scaled = scaler_X.transform(last_df[feature_cols])
                temp_sequence[0] = new_scaled
            except Exception as e:
                print(f"更新序列时出错: {e}")
                # 使用最后有效序列继续预测
                break

            current_date = next_date

    # 保存到数据库
    try:
        for date, pred in zip(future_dates, future_preds):
            uncertainty = pred * 0.03  # 3%的不确定性
            existing_forecast = Forecast.query.filter_by(date=date.date()).first()

            if not existing_forecast:
                forecast = Forecast(
                    date=date.date(),
                    yhat=pred,
                    yhat_lower=pred - uncertainty,
                    yhat_upper=pred + uncertainty
                )
                db.session.add(forecast)
            else:
                existing_forecast.yhat = pred
                existing_forecast.yhat_lower = pred - uncertainty
                existing_forecast.yhat_upper = pred + uncertainty

        db.session.commit()
        print("预测数据已保存到数据库")

    except Exception as e:
        print(f"保存到数据库失败: {e}")
        db.session.rollback()

    # 打印预测结果
    print("\n未来5日预测结果:")
    print("日期        预测价格    置信区间")
    for date, pred in zip(future_dates, future_preds):
        uncertainty = pred * 0.03
        print(f"{date.strftime('%Y-%m-%d')}  {pred:8.2f}  [{pred - uncertainty:.2f}, {pred + uncertainty:.2f}]")

    return {
        'dates': [date.strftime('%Y-%m-%d') for date in future_dates],
        'predictions': future_preds,
        'confidence_intervals': [(pred - pred * 0.03, pred + pred * 0.03) for pred in future_preds]
    }


if __name__ == "__main__":
    from app import app

    with app.app_context():
        result = predict_future()
        if result:
            print("\n预测完成！")
        else:
            print("\n预测失败！")