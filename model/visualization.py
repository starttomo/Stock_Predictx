# model/visualization.py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch
import os
import matplotlib.font_manager as fm
from datetime import datetime, timedelta


def setup_chinese_fonts():
    """设置中文字体，确保中文正常显示"""
    try:
        # 方法1: 使用项目根目录的SimHei.ttf文件
        current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        font_path = os.path.join(current_dir, 'SimHei.ttf')

        # 如果上述路径找不到，尝试当前目录
        if not os.path.exists(font_path):
            current_dir = os.path.dirname(os.path.abspath(__file__))
            font_path = os.path.join(current_dir, 'SimHei.ttf')

        # 如果找到字体文件，注册并使用
        if os.path.exists(font_path):
            # 清除字体缓存并重新加载
            fm._rebuild()

            # 注册字体
            fm.fontManager.addfont(font_path)
            font_prop = fm.FontProperties(fname=font_path)
            font_name = font_prop.get_name()

            # 设置全局字体
            plt.rcParams['font.family'] = font_name
            plt.rcParams['font.sans-serif'] = [font_name, 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False

            print(f"成功加载本地中文字体: {font_name}")

            # 验证字体设置
            from matplotlib import font_manager
            available_fonts = [f.name for f in font_manager.fontManager.ttflist if font_name.lower() in f.name.lower()]
            print(f"可用字体中包含: {available_fonts}")

        else:
            # 如果本地字体文件不存在，回退到系统字体
            print(f"未找到本地字体文件: {font_path}")
            setup_system_fonts()

    except Exception as e:
        print(f"字体设置出错: {e}")
        # 出错时回退到系统字体
        setup_system_fonts()


def setup_system_fonts():
    """设置系统中文字体（备用方案）"""
    # 尝试多种常见中文字体
    font_candidates = [
        'SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei',
        'Heiti TC', 'Arial Unicode MS', 'DejaVu Sans'
    ]

    # 查找系统中可用的中文字体
    available_fonts = []
    for font_name in font_candidates:
        try:
            # 检查字体是否可用
            if any(font_name in f.name for f in fm.fontManager.ttflist):
                available_fonts.append(font_name)
        except:
            continue

    if available_fonts:
        plt.rcParams['font.family'] = available_fonts
        plt.rcParams['font.sans-serif'] = available_fonts
        plt.rcParams['axes.unicode_minus'] = False
        print(f"使用系统中文字体: {available_fonts}")
    else:
        plt.rcParams['font.family'] = ['sans-serif']
        print("警告: 未找到中文字体，可能无法正常显示中文")


# 初始化字体设置
setup_chinese_fonts()


class ModelVisualizer:
    """LSTM模型可视化工具类"""

    def __init__(self, save_dir='model/plots'):
        """
        初始化可视化器
        Args:
            save_dir: 图表保存目录
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.fig_count = 0

    def plot_training_history(self, train_losses, val_losses, lrs=None,
                              save_path=None, show=False):
        """绘制训练过程曲线"""
        # 创建新图形时再次确认字体设置
        plt.rcParams["font.family"] = plt.rcParams["font.family"]

        fig, axes = plt.subplots(2, 1, figsize=(12, 10))

        # 损失曲线
        axes[0].plot(train_losses, label='训练损失', linewidth=2, alpha=0.8)
        axes[0].plot(val_losses, label='验证损失', linewidth=2, alpha=0.8)
        axes[0].set_title('模型训练损失曲线', fontsize=16, fontweight='bold')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # 添加最佳epoch标注
        min_val_loss = min(val_losses)
        best_epoch = val_losses.index(min_val_loss)
        axes[0].axvline(x=best_epoch, color='r', linestyle='--', alpha=0.5)
        axes[0].text(best_epoch, min_val_loss, f'最佳轮次: {best_epoch}\nVal Loss: {min_val_loss:.6f}',
                     ha='center', va='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # 学习率曲线
        if lrs is not None:
            axes[1].plot(lrs, label='学习率', color='green', linewidth=2)
            axes[1].set_title('学习率变化曲线', fontsize=16, fontweight='bold')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Learning Rate')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            axes[1].set_yscale('log')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"训练历史图已保存至: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def plot_predictions_vs_actual(self, y_true, y_pred, dates=None,
                                   save_path=None, show=False):
        """绘制预测值与真实值对比图"""
        plt.rcParams["font.family"] = plt.rcParams["font.family"]

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 1. 时间序列对比图
        if dates is not None:
            axes[0, 0].plot(dates, y_true, label='真实价格', linewidth=2, alpha=0.8)
            axes[0, 0].plot(dates, y_pred, label='预测价格', linewidth=2, alpha=0.8)
        else:
            axes[0, 0].plot(y_true, label='真实价格', linewidth=2, alpha=0.8)
            axes[0, 0].plot(y_pred, label='预测价格', linewidth=2, alpha=0.8)
        axes[0, 0].set_title('预测值 vs 真实值（时间序列）', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('价格')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. 散点图
        axes[0, 1].scatter(y_true, y_pred, alpha=0.6, s=20)
        axes[0, 1].plot([y_true.min(), y_true.max()],
                        [y_true.min(), y_true.max()],
                        'r--', lw=2, label='完美预测线')
        axes[0, 1].set_title('预测值 vs 真实值（散点图）', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('真实价格')
        axes[0, 1].set_ylabel('预测价格')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 计算R²
        r2 = r2_score(y_true, y_pred)
        axes[0, 1].text(0.05, 0.95, f'R² = {r2:.4f}',
                        transform=axes[0, 1].transAxes,
                        fontsize=12, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # 3. 残差图
        residuals = y_true - y_pred
        axes[1, 0].scatter(y_pred, residuals, alpha=0.6, s=20)
        axes[1, 0].axhline(y=0, color='r', linestyle='--', linewidth=2)
        axes[1, 0].set_title('残差分析图', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('预测价格')
        axes[1, 0].set_ylabel('残差（真实-预测）')
        axes[1, 0].grid(True, alpha=0.3)

        # 4. 残差分布直方图
        axes[1, 1].hist(residuals, bins=50, alpha=0.7, edgecolor='black')
        axes[1, 1].set_title('残差分布直方图', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('残差')
        axes[1, 1].set_ylabel('频数')
        axes[1, 1].grid(True, alpha=0.3)

        # 添加残差统计信息
        mean_resid = np.mean(residuals)
        std_resid = np.std(residuals)
        axes[1, 1].text(0.05, 0.95, f'均值: {mean_resid:.4f}\n标准差: {std_resid:.4f}',
                        transform=axes[1, 1].transAxes,
                        fontsize=12, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"预测对比图已保存至: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

        return residuals

    def plot_direction_accuracy(self, y_true, y_pred, dates=None,
                                save_path=None, show=False):
        """绘制方向准确率分析图"""
        plt.rcParams["font.family"] = plt.rcParams["font.family"]

        # 计算涨跌方向
        true_directions = np.diff(y_true.flatten()) > 0
        pred_directions = np.diff(y_pred.flatten()) > 0
        direction_correct = true_directions == pred_directions

        fig, axes = plt.subplots(2, 1, figsize=(14, 10))

        # 1. 价格与方向对比
        if dates is not None:
            axes[0].plot(dates, y_true, label='真实价格', linewidth=2, alpha=0.8)
            axes[0].plot(dates, y_pred, label='预测价格', linewidth=2, alpha=0.8)

            # 标记正确和错误的预测点
            correct_dates = dates[1:][direction_correct]
            correct_prices = y_true.flatten()[1:][direction_correct]
            wrong_dates = dates[1:][~direction_correct]
            wrong_prices = y_true.flatten()[1:][~direction_correct]

            axes[0].scatter(correct_dates, correct_prices, c='green', s=30,
                            label='方向正确', alpha=0.7, marker='^')
            axes[0].scatter(wrong_dates, wrong_prices, c='red', s=30,
                            label='方向错误', alpha=0.7, marker='v')
        else:
            axes[0].plot(y_true, label='真实价格', linewidth=2, alpha=0.8)
            axes[0].plot(y_pred, label='预测价格', linewidth=2, alpha=0.8)

        axes[0].set_title('价格走势与方向预测结果', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('价格')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # 2. 方向准确率柱状图
        correct_count = np.sum(direction_correct)
        wrong_count = len(direction_correct) - correct_count
        accuracy = correct_count / len(direction_correct)

        axes[1].bar(['正确', '错误'], [correct_count, wrong_count],
                    color=['green', 'red'], alpha=0.7)
        axes[1].set_title(f'方向预测准确率: {accuracy:.2%}', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('预测次数')

        # 添加数值标签
        for i, v in enumerate([correct_count, wrong_count]):
            axes[1].text(i, v + max(correct_count, wrong_count) * 0.01,
                         str(v), ha='center', va='bottom', fontweight='bold')

        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"方向准确率图已保存至: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

        return accuracy

    def plot_future_forecast(self, historical_dates, historical_prices,
                             forecast_dates, forecast_prices,
                             forecast_lower=None, forecast_upper=None,
                             save_path=None, show=False):
        """绘制未来预测效果图"""
        plt.rcParams["font.family"] = plt.rcParams["font.family"]

        fig, ax = plt.subplots(figsize=(14, 8))

        # 绘制历史数据
        ax.plot(historical_dates, historical_prices,
                label='历史价格', linewidth=2, color='blue', alpha=0.8)

        # 绘制预测数据
        ax.plot(forecast_dates, forecast_prices,
                label='预测价格', linewidth=2, color='orange', alpha=0.8)

        # 绘制预测区间
        if forecast_lower is not None and forecast_upper is not None:
            ax.fill_between(forecast_dates, forecast_lower, forecast_upper,
                            alpha=0.3, color='orange', label='预测区间')

        # 添加分界线和标注
        last_date = historical_dates[-1]
        ax.axvline(x=last_date, color='red', linestyle='--', alpha=0.7, linewidth=2)
        ax.text(last_date, ax.get_ylim()[1] * 0.95, '预测起点',
                rotation=90, ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

        ax.set_title('沪深300价格预测效果图', fontsize=16, fontweight='bold')
        ax.set_xlabel('日期')
        ax.set_ylabel('价格')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 旋转日期标签
        plt.xticks(rotation=45)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"未来预测图已保存至: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def plot_feature_correlation(self, df, feature_cols, target_col='close',
                                 save_path=None, show=False):
        """绘制特征相关性热力图"""
        plt.rcParams["font.family"] = plt.rcParams["font.family"]

        # 计算相关性矩阵
        corr_matrix = df[feature_cols + [target_col]].corr()

        fig, ax = plt.subplots(figsize=(14, 12))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                    center=0, square=True, ax=ax, cbar_kws={"shrink": 0.8})
        ax.set_title('特征相关性热力图', fontsize=16, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"特征相关性图已保存至: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

        return corr_matrix


# 便捷的独立函数
def create_training_plots(train_losses, val_losses, save_dir='model/plots'):
    """快速创建训练过程图"""
    visualizer = ModelVisualizer(save_dir)
    visualizer.plot_training_history(
        train_losses, val_losses,
        save_path=f"{save_dir}/training_history.png",
        show=False
    )


def create_evaluation_plots(y_true, y_pred, dates=None, save_dir='model/plots'):
    """快速创建评估图"""
    visualizer = ModelVisualizer(save_dir)
    visualizer.plot_predictions_vs_actual(
        y_true, y_pred, dates,
        save_path=f"{save_dir}/evaluation.png",
        show=False
    )
    visualizer.plot_direction_accuracy(
        y_true, y_pred, dates,
        save_path=f"{save_dir}/direction_accuracy.png",
        show=False
    )


def create_forecast_plots(historical_data, forecast_data, save_dir='model/plots'):
    """快速创建预测图"""
    visualizer = ModelVisualizer(save_dir)
    visualizer.plot_future_forecast(
        historical_data['dates'],
        historical_data['prices'],
        forecast_data['dates'],
        forecast_data['prices'],
        forecast_data.get('lower'),
        forecast_data.get('upper'),
        save_path=f"{save_dir}/future_forecast.png",
        show=False
    )