import matplotlib.pyplot as plt
from metrics_data2 import *

# --- 数据定义（已迁移至 metrics_data.py）---


# --- 绘图 ---
plt.figure() # 创建一个新的图形窗口 (MATLAB: figure)
# 'hold on' 在 matplotlib 中是默认行为，多次 plot 会绘制在同一张图上

# 要绘制的指标
show = 'acc' 
markersize = 4

model_name = [
    # [Fcnn_tiny_metrics, 'Fcnn(tiny)'],
    # [Tatt_tiny_metrics, 'Tatt(tiny)'],
    [F_T_tiny2_metrics, 'F_T(tiny2)'],
    [F_T_mini2_metrics, 'F_T(mini2)'],
    [F_T_mini2_gate_mulitloss_metrics, 'F_T(mini2-gate-multi-loss)'],
    # [S3RImageNet_TinyTransformer_metrics, 'yll'],

    # [S3R_metrics, 'S3R'],
    # [MobileNetV3Small_ETF_metrics, 'MobileNetV3'],
    # [resnet18_metrics, 'resnet18']
]


for metics, name in model_name:
    plt.plot(SNR, metics[show], '-v', linewidth=1.5, markersize=markersize, label=name)


# 'hold off' 在这里不是必需的

# --- 格式设置 ---
plt.xlabel('SNR (dB)', fontsize=12) # X轴标签
plt.ylabel('Avg_acc (%)', fontsize=12) # Y轴标签 (MATLAB的 'Avg\_acc' 在此写作 'Avg_acc')
plt.grid(True) # 显示网格线 (MATLAB: grid on)

# 设置坐标轴刻度字体大小 (MATLAB: ax.FontSize = 11)
plt.tick_params(axis='both', which='major', labelsize=11)

# 显示图例并设置位置 (MATLAB: legend('show', 'Location', 'northwest'))
plt.legend(loc='upper left') 

# --- 显示图形 ---
plt.savefig('performance_comparison.png', dpi=300) # 保存图形为高分辨率PNG文件