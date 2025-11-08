# model/evaluate.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import torch
import joblib

# 添加MAPE和Sharpe Ratio：计算机验证模型准确，经济学上评估策略回报/风险（Markowitz, 1952）。

def evaluate_model(model, X_test, y_test, scaler_y):
    """评估模型性能"""
    model.eval()
    with torch.no_grad():
        predictions_scaled = model(torch.FloatTensor(X_test)).numpy()

    # 反标准化（y_test 是缩放后的，需要反缩放）
    predictions = scaler_y.inverse_transform(predictions_scaled)
    y_true = scaler_y.inverse_transform(y_test.reshape(-1, 1))

    mae = mean_absolute_error(y_true, predictions)
    mse = mean_squared_error(y_true, predictions)
    rmse = np.sqrt(mse)

    # MAPE (百分比误差)
    mape = np.mean(np.abs((y_true - predictions) / y_true)) * 100 if np.all(y_true != 0) else float('inf')

    # 方向准确率
    direction_true = np.diff(y_true.flatten()) > 0
    direction_pred = np.diff(predictions.flatten()) > 0
    direction_accuracy = np.mean(direction_true == direction_pred)

    # Sharpe Ratio (简化：假设无风险率为0，用预测回报计算)
    returns_pred = np.diff(predictions.flatten()) / predictions.flatten()[:-1]
    sharpe = np.mean(returns_pred) / np.std(returns_pred) if np.std(returns_pred) != 0 else 0

    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAPE: {mape:.2f}%")
    print(f"方向准确率: {direction_accuracy:.2%}")
    print(f"Sharpe Ratio (预测): {sharpe:.4f}")

    return mae, rmse, mape, direction_accuracy, sharpe