# model/train.py - 改进版本（含可视化）
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from data.loader import get_hs300_data
from model.lstm_model import ImprovedLSTMModel
from database.models import db, Forecast
from datetime import datetime, timedelta
import joblib
import os
import matplotlib.pyplot as plt

# 导入可视化模块
from model.visualization import ModelVisualizer, create_training_plots, create_evaluation_plots, create_forecast_plots


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
    print("=" * 60)
    print("开始改进版模型训练...")
    print("=" * 60)

    # 初始化可视化器
    visualizer = ModelVisualizer(save_dir='model/plots')

    # 获取数据并创建特征
    print("Step 1: 加载数据并创建特征...")
    df = get_hs300_data()
    df = create_features(df)
    feature_cols = [col for col in df.columns if col != 'date']
    print(f"✓ 特征工程完成，特征数量: {len(feature_cols)}")
    print(f"✓ 特征列表: {', '.join(feature_cols[:10])}{'...' if len(feature_cols) > 10 else ''}")

    # 数据标准化（先缩放所有特征，包括 close）
    print("\nStep 2: 数据标准化...")
    scaler_X = MinMaxScaler()
    data_scaled = scaler_X.fit_transform(df[feature_cols])
    scaled_df = pd.DataFrame(data_scaled, columns=feature_cols, index=df.index)
    scaled_df['date'] = df['date']

    # 单独的 scaler_y 只用于 close（便于反缩放预测）
    scaler_y = MinMaxScaler()
    scaler_y.fit(df[['close']])  # 只 fit 原始 close
    print("✓ 标准化完成")

    # 准备序列数据（使用缩放后的 df）
    print("\nStep 3: 准备序列数据...")
    seq_len = 60
    X, y = prepare_sequences_multivariate(scaled_df, seq_len)

    if len(X) == 0:
        print("错误：没有足够的数据生成序列")
        return

    print(f"✓ 数据形状: X={X.shape}, y={y.shape}")

    # 分割数据
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # 创建改进的LSTM模型
    print("\nStep 4: 创建模型...")
    input_size = X.shape[2]  # 现在是所有特征的数量
    model = ImprovedLSTMModel(input_size=input_size, hidden_size=100, num_layers=3, output_size=1)
    print(f"✓ 模型结构: {model}")

    # 训练配置
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    # 记录训练历史
    train_losses = []
    val_losses = []
    lrs = []
    best_loss = float('inf')
    patience = 40
    patience_counter = 0

    print("\nStep 5: 开始训练...")
    print("=" * 40)

    # 训练循环
    for epoch in range(200):
        model.train()
        train_loss = 0
        num_batches = 0

        for i in range(0, len(X_train), 32):
            batch_x = torch.FloatTensor(X_train[i:i + 32])
            batch_y = torch.FloatTensor(y_train[i:i + 32].reshape(-1, 1))

            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
            num_batches += 1

        # 验证
        model.eval()
        with torch.no_grad():
            val_pred = model(torch.FloatTensor(X_test))
            val_loss = criterion(val_pred, torch.FloatTensor(y_test.reshape(-1, 1)))

        # 记录历史
        train_losses.append(train_loss / num_batches)
        val_losses.append(val_loss.item())
        lrs.append(optimizer.param_groups[0]['lr'])

        scheduler.step(val_loss)

        if epoch % 20 == 0:
            print(f"Epoch {epoch:3d} | Train Loss: {train_loss / num_batches:.6f} | "
                  f"Val Loss: {val_loss:.6f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

        # 早停
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'model/best_model.pth')
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"→ 早停于第 {epoch} 轮")
            break

    print("=" * 40)
    print(f"✓ 训练完成！最佳验证损失: {best_loss:.6f}")

    # 加载最佳模型
    model.load_state_dict(torch.load('model/best_model.pth'))

    # 保存scaler
    os.makedirs('model', exist_ok=True)
    joblib.dump(scaler_X, 'model/scaler_X.pkl')
    joblib.dump(scaler_y, 'model/scaler_y.pkl')
    print("✓ 模型和Scaler已保存")

    # === 可视化部分 ===
    print("\n" + "=" * 60)
    print("生成可视化图表...")
    print("=" * 60)

    # 1. 训练过程可视化
    print("1. 生成训练历史图表...")
    visualizer.plot_training_history(
        train_losses, val_losses, lrs,
        save_path='model/plots/training_history.png',
        show=False
    )

    # 2. 模型评估可视化
    print("2. 生成模型评估图表...")
    model.eval()
    with torch.no_grad():
        test_predictions_scaled = model(torch.FloatTensor(X_test)).numpy()

    test_predictions = scaler_y.inverse_transform(test_predictions_scaled)
    y_test_true = scaler_y.inverse_transform(y_test.reshape(-1, 1))
    test_dates = df['date'].iloc[split + seq_len:].values

    # 评估指标
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    mae = mean_absolute_error(y_test_true, test_predictions)
    rmse = np.sqrt(mean_squared_error(y_test_true, test_predictions))

    print(f"   - MAE: {mae:.2f}")
    print(f"   - RMSE: {rmse:.2f}")

    # 生成评估图表
    visualizer.plot_predictions_vs_actual(
        y_test_true, test_predictions, test_dates,
        save_path='model/plots/evaluation.png',
        show=False
    )

    visualizer.plot_direction_accuracy(
        y_test_true, test_predictions, test_dates,
        save_path='model/plots/direction_accuracy.png',
        show=False
    )

    # 3. 特征相关性可视化
    print("3. 生成特征相关性图表...")
    visualizer.plot_feature_correlation(
        df, feature_cols,
        save_path='model/plots/feature_correlation.png',
        show=False
    )

    # 4. 预测未来并可视化
    print("\nStep 6: 预测未来5日价格...")
    model.eval()
    last_sequence = X[-1:].copy()

    future_dates = []
    future_preds = []
    current_date = df['date'].iloc[-1]

    with torch.no_grad():
        temp_sequence = last_sequence.copy()
        print("\n预测进度:")

        for i in range(5):
            next_date = current_date + timedelta(days=1)
            while next_date.weekday() >= 5:  # 跳过周末
                next_date += timedelta(days=1)

            # 预测
            pred_scaled = model(torch.FloatTensor(temp_sequence)).numpy()[0, 0]
            pred_price = scaler_y.inverse_transform([[pred_scaled]])[0, 0]

            future_preds.append(pred_price)
            future_dates.append(next_date)

            # 更新序列
            new_row = temp_sequence[0, -1, :].copy()
            new_row[0] = pred_scaled
            temp_sequence = np.roll(temp_sequence, -1, axis=1)
            temp_sequence[0, -1, :] = new_row

            current_date = next_date
            print(f"   - {next_date.strftime('%Y-%m-%d')}: {pred_price:.2f}")

    # 生成预测效果图
    print("\n4. 生成未来预测图表...")
    historical_dates = df['date'].iloc[-60:].values  # 最近60天
    historical_prices = df['close'].iloc[-60:].values

    forecast_lower = [p - p * 0.03 for p in future_preds]
    forecast_upper = [p + p * 0.03 for p in future_preds]

    visualizer.plot_future_forecast(
        historical_dates, historical_prices,
        future_dates, future_preds,
        forecast_lower, forecast_upper,
        save_path='model/plots/future_forecast.png',
        show=False
    )

    # 存入数据库
    print("\nStep 7: 保存预测结果到数据库...")
    for date, pred in zip(future_dates, future_preds):
        uncertainty = pred * 0.03
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

    print("\n" + "=" * 60)
    print("所有任务完成！图表已保存至 model/plots/ 目录")
    print("=" * 60)

    # 打印总结报告
    print("\n📊 训练总结报告:")
    print(f"   - 训练轮次: {len(train_losses)}")
    print(f"   - 最佳验证损失: {best_loss:.6f}")
    print(f"   - 最终训练损失: {train_losses[-1]:.6f}")
    print(f"   - 测试集 MAE: {mae:.2f}")
    print(f"   - 测试集 RMSE: {rmse:.2f}")
    print(f"\n   - 图表位置:")
    print(f"     * 训练历史: model/plots/training_history.png")
    print(f"     * 评估结果: model/plots/evaluation.png")
    print(f"     * 方向准确率: model/plots/direction_accuracy.png")
    print(f"     * 特征相关性: model/plots/feature_correlation.png")
    print(f"     * 未来预测: model/plots/future_forecast.png")


if __name__ == "__main__":
    from app import app

    with app.app_context():
        train_and_forecast_improved()