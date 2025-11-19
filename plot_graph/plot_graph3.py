import matplotlib.pyplot as plt
from metrics_data3 import *

# --- 数据定义（已迁移至 metrics_data.py）---


# --- 绘图 ---
plt.figure() # 创建一个新的图形窗口 (MATLAB: figure)
# 'hold on' 在 matplotlib 中是默认行为，多次 plot 会绘制在同一张图上

# 要绘制的指标
show = 'acc' 
markersize = 4

model_name = [
    [F_cnn_nano_1119_metrics, 'HARNet(nano)'],
    [F_cnn_mini_1119_metrics, 'HARNet(mini)'],
    [F_cnn_tiny_1119_metrics, 'HARNet(tiny)'],
    [F_cnn_small_1119_metrics, 'HARNet(small)'],
    [F_cnn_base_1119_metrics, 'HARNet(base)'],
    [F_cnn_large_1119_metrics, 'HARNet(large)'],
    [T_att_nano_1119_metrics, 'SSTA(nano)'],
    [T_att_mini_1119_metrics, 'SSTA(mini)'],
    [T_att_tiny_1119_metrics, 'SSTA(tiny)'],
    [T_att_small_1119_metrics, 'SSTA(small)'],
    [T_att_base_1119_metrics, 'SSTA(base)'],
    [T_att_large_1119_metrics, 'SSTA(large)'],
    
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