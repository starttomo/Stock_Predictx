# model/predict.py
import torch
import joblib
import numpy as np
from model.lstm_model import ImprovedLSTMModel  # 改为改进模型

# 加载模型和 scaler
model = ImprovedLSTMModel(input_size=33)  # 调整为你的特征数（33）
model.load_state_dict(torch.load('model/best_model.pth', map_location='cpu'))  # 用 best_model
model.eval()
scaler_X = joblib.load('model/scaler_X.pkl')
scaler_y = joblib.load('model/scaler_y.pkl')

def predict_future(last_60_features):
    # 假设 last_60_features 是 (60, num_features) 的原始特征数组
    scaled = scaler_X.transform(last_60_features.reshape(60, -1))  # 缩放
    X = torch.FloatTensor(scaled).reshape(1, 60, -1)
    with torch.no_grad():
        pred_scaled = model(X).numpy()
    return scaler_y.inverse_transform(pred_scaled)[0][0]