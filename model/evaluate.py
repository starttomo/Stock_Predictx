# model/evaluate.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import torch
import joblib

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

    # 方向准确率
    direction_true = np.diff(y_true.flatten()) > 0
    direction_pred = np.diff(predictions.flatten()) > 0
    direction_accuracy = np.mean(direction_true == direction_pred)

    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"方向准确率: {direction_accuracy:.2%}")

    return mae, rmse, direction_accuracy