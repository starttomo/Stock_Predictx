# utils/data_augmentation.py
import numpy as np
import pandas as pd


def augment_data(df, num_augmented=5):
    """数据增强"""
    augmented_dfs = []

    for i in range(num_augmented):
        aug_df = df.copy()

        # 添加小幅随机噪声
        noise = np.random.normal(0, 0.01, len(df))
        aug_df['close'] = aug_df['close'] * (1 + noise)

        # 轻微的时间偏移
        if len(df) > 10:
            shift = np.random.randint(-3, 3)
            aug_df['close'] = aug_df['close'].shift(shift).fillna(method='bfill')

        augmented_dfs.append(aug_df)

    return pd.concat([df] + augmented_dfs, ignore_index=True)


def get_more_data():
    """获取更多历史数据"""
    # 如果可以的话，获取更长时间范围的数据
    # 这里可以添加从其他数据源获取数据的逻辑
    pass