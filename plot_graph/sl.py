import pandas as pd
import numpy as np

def analyze_convergence(file_path):
    # 读取CSV文件 (假设分隔符是制表符 \t，如果是逗号请改为 sep=',')
    try:
        # 根据你提供的数据格式，看起来是制表符分隔
        df = pd.read_csv(file_path, sep='\t')
    except:
        # 如果读取失败，尝试用逗号分隔
        df = pd.read_csv(file_path, sep=',')

    # 确保列名没有空格干扰
    df.columns = [c.strip() for c in df.columns]
    
    # 1. 方法一：基于最大值 (Max Value) - 最常用的"最佳模型"判断
    # 找到Value列最大值所在的行
    best_epoch_row = df.loc[df['Value'].idxmax()]
    best_step = int(best_epoch_row['Step'])
    best_val = best_epoch_row['Value']
    
    print(f"【基于最高精度判断】")
    print(f"验证集 Acc 在第 {best_step} 轮达到最高值: {best_val:.6f}")
    print("-" * 30)

    # 2. 方法二：基于滑动窗口的稳定性 (Stability)
    # 比如：检查最近3轮的标准差是否小于某个阈值 (例如 0.005)
    window_size = 3
    threshold = 0.005
    
    # 计算滑动窗口标准差
    df['rolling_std'] = df['Value'].rolling(window=window_size).std()
    
    # 找到第一个满足条件的点
    stable_points = df[df['rolling_std'] < threshold]
    
    print(f"【基于平稳性判断 (窗口={window_size}, 阈值={threshold})】")
    if not stable_points.empty:
        first_stable_step = int(stable_points.iloc[0]['Step'])
        first_stable_val = stable_points.iloc[0]['Value']
        print(f"验证集 Acc 在第 {first_stable_step} 轮开始趋于平稳 (方差变小)，此时值为: {first_stable_val:.6f}")
        print("注意：平稳不代表最高，可能陷入局部最优或平台期。")
    else:
        print("在该数据范围内，未检测到明显的平稳收敛（变化幅度仍较大）。")

# --- 模拟运行你的数据 ---
# 你可以直接替换下面的 csv_content 为你的文件路径，例如: analyze_convergence('your_file.csv')

import io

import csv
# 打开CSV文件
with open('example.csv', newline='') as csvfile:
    csv_content = csv.reader(csvfile, delimiter=',')

# 创建一个虚拟文件进行演示
dummy_file = io.StringIO(csv_content)

# 运行分析
analyze_convergence(dummy_file)