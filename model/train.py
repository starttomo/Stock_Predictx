# model/train.py - 改进版本
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from data.loader import get_hs300_data
from model.lstm_model import ImprovedLSTMModel  # 改为导入改进模型
from database.models import db, Forecast
from datetime import datetime, timedelta
import joblib
import os

# 计算机原理：LSTM通过门控机制捕捉时间序列长期依赖（Hochreiter & Schmidhuber, 1997）。
# 多步预测用自回归更新所有特征，避免误差累积（参考arXiv 2020 "Stock Price Prediction Using Machine Learning and LSTM"）。

def create_features(df, is_predict=False):
        close = df['close']

        # 价格特征
        df['returns'] = close.pct_change()
        df['log_returns'] = np.log(close / close.shift(1))

        # 移动平均线 - 预测时用min(window, len(df))避免NaN
        for window in [5, 10, 20, 30, 60]:
            roll_win = min(window, len(df)) if is_predict else window
            df[f'ma{window}'] = close.rolling(roll_win).mean()
            df[f'ma_ratio_{window}'] = close / df[f'ma{window}']

        # 波动率 - 同上
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

        # MACD - ewm不需window调整，但初始NaN少
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

        # 强制填充所有NaN，不dropna
        return df.ffill().bfill().fillna(0)
def prepare_sequences_multivariate(df, seq_len, target_col='close'):
    """准备多变量序列数据，使用所有可用特征"""
    # 动态获取特征列（除 'date'）
    feature_cols = [col for col in df.columns if col != 'date']
    features = df[feature_cols].values
    targets = df[target_col].values  # 注意：如果 df 已缩放，targets 也是缩放后的

    X, y = [], []
    for i in range(len(features) - seq_len):
        X.append(features[i:i + seq_len])
        y.append(targets[i + seq_len])

    return np.array(X), np.array(y)

def train_and_forecast_improved():
    print("开始改进版模型训练...")

    # 获取数据并创建特征
    df = get_hs300_data()
    df['date'] = pd.to_datetime(df['date'])  # 确保date是datetime
    df = df[df['date'] > '2020-01-01'].reset_index(drop=True)  # 限近期数据，避免历史低价
    print(f"过滤后数据行数: {len(df)}")

    df = create_features(df)
    print(f"特征工程完成，特征数量: {len([col for col in df.columns if col != 'date'])}")

    # 数据标准化
    feature_cols = [col for col in df.columns if col != 'date']
    scaler_X = MinMaxScaler()
    data_scaled = scaler_X.fit_transform(df[feature_cols])
    scaled_df = pd.DataFrame(data_scaled, columns=feature_cols, index=df.index)
    scaled_df['date'] = df['date']

    scaler_y = MinMaxScaler()
    scaler_y.fit(df[['close']])  # fit 原 close

    # 准备序列
    seq_len = 60
    X, y = prepare_sequences_multivariate(scaled_df, seq_len)

    if len(X) == 0:
        print("错误：没有足够的数据生成序列")
        return

    print(f"数据形状: X={X.shape}, y={y.shape}")

    # 分割
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # 模型
    input_size = X.shape[2]
    model = ImprovedLSTMModel(input_size=input_size, hidden_size=100, num_layers=3, output_size=1)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)  # 调整lr
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    # 训练
    best_loss = float('inf')
    patience = 50  # 增大
    patience_counter = 0

    for epoch in range(200):
        model.train()
        train_loss = 0
        for i in range(0, len(X_train), 32):
            batch_x = torch.FloatTensor(X_train[i:i + 32])
            batch_y = torch.FloatTensor(y_train[i:i + 32].reshape(-1, 1))  # 缩放 y

            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        # 验证
        model.eval()
        with torch.no_grad():
            val_pred = model(torch.FloatTensor(X_test))
            val_loss = criterion(val_pred, torch.FloatTensor(y_test.reshape(-1, 1)))

        scheduler.step(val_loss)

        if epoch % 20 == 0:
            print(f"Epoch {epoch}, Train Loss: {train_loss / (len(X_train)/32):.6f}, Val Loss: {val_loss:.6f}")  # 修正 Train Loss 计算（除以 batch 数）

        # 早停
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'model/best_model.pth')
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"早停于第 {epoch} 轮")
            break

    # 加载最佳模型
    model.load_state_dict(torch.load('model/best_model.pth'))

    # 保存 scaler
    os.makedirs('model', exist_ok=True)
    joblib.dump(scaler_X, 'model/scaler_X.pkl')
    joblib.dump(scaler_y, 'model/scaler_y.pkl')

    # 预测未来 - 改进：完整更新所有特征
    model.eval()
    last_sequence = X[-1:].copy()  # 缩放后的最后序列 (1, 60, num_features)
    last_df = df.tail(252).copy()  # 增大历史，避NaN

    future_dates = []
    future_preds = []
    current_date = df['date'].iloc[-1]

    with torch.no_grad():
        temp_sequence = last_sequence.copy()

        for i in range(5):
            next_date = current_date + timedelta(days=1)
            while next_date.weekday() >= 5:  # 跳过周末
                next_date += timedelta(days=1)

            # 预测（缩放尺度）
            pred_scaled = model(torch.FloatTensor(temp_sequence)).numpy()[0, 0]

            # 反缩放得到真实价格
            pred_price = scaler_y.inverse_transform([[pred_scaled]])[0, 0]

            future_preds.append(pred_price)
            future_dates.append(next_date)

            # 更新原始df：添加新行（close=pred_price，其他如volume用最后均值简化）
            new_row = last_df.iloc[-1].copy()
            new_row['date'] = next_date
            new_row['close'] = pred_price
            new_row['open'] = pred_price  # 简化
            new_row['high'] = pred_price * 1.01
            new_row['low'] = pred_price * 0.99
            new_row['volume'] = last_df['volume'].mean()  # 均值填充

            last_df = pd.concat([last_df, pd.DataFrame([new_row])], ignore_index=True)
            last_df = create_features(last_df, is_predict=True).tail(seq_len)  # 修复is_predict=True

            # 缩放新序列 - 加try防空
            try:
                new_scaled = scaler_X.transform(last_df[feature_cols])
                temp_sequence[0] = new_scaled  # 更新序列
            except ValueError:
                print("警告: 序列空，使用最后序列")
                pred_price = df['close'].mean()  # fallback

            current_date = next_date

    # 存入数据库
    for date, pred in zip(future_dates, future_preds):
        uncertainty = pred * 0.03  # 3%的不确定性
        f = db.session.get(Forecast, date.date())  # 修复legacy
        if not f:
            f = Forecast(
                date=date.date(),
                yhat=pred,
                yhat_lower=pred - uncertainty,
                yhat_upper=pred + uncertainty
            )
            db.session.add(f)
        else:
            f.yhat = pred
            f.yhat_lower = pred - uncertainty
            f.yhat_upper = pred + uncertainty

    db.session.commit()

    print("预测完成！未来5日预测:")
    for date, pred in zip(future_dates, future_preds):
        print(f"{date.strftime('%Y-%m-%d')}: {pred:.2f}")

if __name__ == "__main__":
    from app import app
    with app.app_context():
        train_and_forecast_improved()