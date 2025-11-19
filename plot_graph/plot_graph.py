import matplotlib.pyplot as plt
from metrics_data import *

# --- 数据定义（已迁移至 metrics_data.py）---


# --- 绘图 ---
plt.figure() # 创建一个新的图形窗口 (MATLAB: figure)
# 'hold on' 在 matplotlib 中是默认行为，多次 plot 会绘制在同一张图上

# 要绘制的指标
show = 'acc' 
markersize = 4

###A分支以及变种模型

# plt.plot(SNR, Res_Inc_Net_tiny_metrics[show], '-x', linewidth=1.5, markersize=markersize, label='ResIncNet(tiny)')
# plt.plot(SNR, Res_Inc_Net_small_metrics[show], '-d', linewidth=1.5, markersize=markersize, label='ResIncNet(small)')
# plt.plot(SNR, Res_Inc_Net_large_metrics[show], '-o', linewidth=1.5, markersize=markersize, label='ResIncNet(large)')
# plt.plot(SNR, Res_Inc_Net_base_metrics[show], '-s', linewidth=1.5, markersize=markersize, label='ResIncNet(base)')

# plt.plot(SNR, SW_Res_Inc_Net_base_metrics[show], '-^', linewidth=1.5, markersize=markersize, label='ResIncNet(SW-base)')
# plt.plot(SNR, SW_Res_Inc_Net_base_ECANet_metrics[show], '-v', linewidth=1.5, markersize=markersize, label='ResIncNet(SW-ECANet-base)')
plt.plot(SNR, SW_Res_Inc_Net_base_SENet_metrics[show], '-v', linewidth=1.5, markersize=markersize, label='ResIncNet(SW-SENet-base)') # best

# plt.plot(SNR, HAR_Net_tiny_metrics[show], '-^', linewidth=1.5, markersize=markersize, label='HAR_Net_tiny')
# plt.plot(SNR, HAR_Net_small_metrics[show], '-^', linewidth=1.5, markersize=markersize, label='HAR_Net_small')
# plt.plot(SNR, HAR_Net_base_metrics[show], '-^', linewidth=1.5, markersize=markersize, label='HAR_Net_small')




###B分支以及变种模型
# plt.plot(SNR, rf_att_large_metrics[show], '-o', linewidth=1.5, markersize=markersize, label='SpectraAtt(large)')
# plt.plot(SNR, rf_att_small_metrics[show], '-d', linewidth=1.5, markersize=markersize, label='SpectraAtt(small)')
# plt.plot(SNR, rf_att_medium_metrics[show], '-s', linewidth=1.5, markersize=markersize, label='SpectraAtt(medium)')
# plt.plot(SNR, rf_att_medium_Hybrid_metrics[show], '-^', linewidth=1.5, markersize=markersize, label='SpectraAtt-Hybrid(medium)') # 再优化一下(该次可不参考了)
# plt.plot(SNR, rf_att_tiny_metrics[show], '-d', linewidth=1.5, markersize=markersize, label='SpectraAtt(tiny)')

# plt.plot(SNR, rf_att_medium_downsample8_metrics[show], '-s', linewidth=1.5, markersize=markersize, label='rf_att_medium_downsample8_metrics')
# plt.plot(SNR, rf_att_medium_downsample_16_metrics[show], '-s', linewidth=1.5, markersize=markersize, label='rf_att_medium_downsample_16_metrics')# best

###A+B分支混合模型
# plt.plot(SNR, ResInc_SpAtt_metrics[show], '-s', linewidth=1.5, markersize=markersize, label='F-T(base-medium)') # best
# plt.plot(SNR, F_T_metrics[show], '-s', linewidth=1.5, markersize=markersize, label='F-T-metrics()')
# plt.plot(SNR, F_T_finetune_metrics[show], '-s', linewidth=1.5, markersize=markersize, label='F-T-metrics(base-medium)')
# plt.plot(SNR, F_T_finetune2_metrics[show], '-s', linewidth=1.5, markersize=markersize, label='F-T-metrics2(base-medium)')


############## 比较几个已有模型的性能 ##############
# plt.plot(SNR, S3R_metrics[show], '-s', linewidth=1.5, markersize=markersize, label='S3R')
# plt.plot(SNR, MobileNetV3Small_ETF_metrics[show], '-s', linewidth=1.5, markersize=markersize, label='MobileNetV3Small(ETF)')
# plt.plot(SNR, resnet18_metrics[show], '-s', linewidth=1.5, markersize=markersize, label='resnet18')

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