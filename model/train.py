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

def create_features(df):
    """创建更多技术指标作为特征"""
    close = df['close']

    # 价格特征
    df['returns'] = close.pct_change()
    df['log_returns'] = np.log(close / close.shift(1))

    # 移动平均线
    for window in [5, 10, 20, 30, 60]:
        df[f'ma{window}'] = close.rolling(window).mean()
        df[f'ma_ratio_{window}'] = close / df[f'ma{window}']

    # 波动率
    df['volatility_5'] = df['returns'].rolling(5).std()
    df['volatility_20'] = df['returns'].rolling(20).std()

    # 动量指标
    df['momentum_5'] = close / close.shift(5) - 1
    df['momentum_10'] = close / close.shift(10) - 1

    # 布林带
    df['bb_middle'] = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    df['bb_upper'] = df['bb_middle'] + 2 * bb_std
    df['bb_lower'] = df['bb_middle'] - 2 * bb_std
    df['bb_position'] = (close - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

    # RSI
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # MACD
    exp1 = close.ewm(span=12).mean()
    exp2 = close.ewm(span=26).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=9).mean()

    # 成交量特征
    df['volume_ma'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma']

    # 价格位置特征
    df['high_52'] = close.rolling(252).max()
    df['low_52'] = close.rolling(252).min()
    df['price_position'] = (close - df['low_52']) / (df['high_52'] - df['low_52'])

    return df.dropna()

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
    df = create_features(df)
    print(f"特征工程完成，特征数量: {len([col for col in df.columns if col != 'date'])}")

    # 数据标准化（先缩放所有特征，包括 close）
    feature_cols = [col for col in df.columns if col != 'date']
    scaler_X = MinMaxScaler()
    data_scaled = scaler_X.fit_transform(df[feature_cols])
    scaled_df = pd.DataFrame(data_scaled, columns=feature_cols, index=df.index)
    scaled_df['date'] = df['date']

    # 单独的 scaler_y 只用于 close（便于反缩放预测）
    scaler_y = MinMaxScaler()
    scaler_y.fit(df[['close']])  # 只 fit 原始 close

    # 准备序列数据（使用缩放后的 df）
    seq_len = 60
    X, y = prepare_sequences_multivariate(scaled_df, seq_len)

    if len(X) == 0:
        print("错误：没有足够的数据生成序列")
        return

    print(f"数据形状: X={X.shape}, y={y.shape}")  # y 现在是缩放后的 close

    # 分割数据
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # 创建改进的LSTM模型
    input_size = X.shape[2]  # 现在是所有特征的数量（33）
    model = ImprovedLSTMModel(input_size=input_size, hidden_size=100, num_layers=3, output_size=1)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    # 训练
    best_loss = float('inf')
    patience = 20
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

    # 预测未来
    model.eval()
    last_sequence = X[-1:].copy()  # 缩放后的最后序列 (1, 60, num_features)

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

            # 更新序列：移位，并用新 pred_scaled 更新 'close' 位置（假设 'close' 是 feature_cols[0]）
            # 其他特征保持最后一行值（简化；实际可重新计算部分如 returns）
            new_row = temp_sequence[0, -1, :].copy()  # 复制最后一行
            new_row[0] = pred_scaled  # 更新 close（假设索引 0 是 close）
            temp_sequence = np.roll(temp_sequence, -1, axis=1)  # 移位
            temp_sequence[0, -1, :] = new_row  # 添加新行

            current_date = next_date

    # 存入数据库
    for date, pred in zip(future_dates, future_preds):
        uncertainty = pred * 0.03  # 3%的不确定性
        f = Forecast.query.get(date.date())
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