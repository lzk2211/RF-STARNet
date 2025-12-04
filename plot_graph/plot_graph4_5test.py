import matplotlib.pyplot as plt
from metrics_data4_5test import *

# 四类指标全部输出并保存 PNG + EPS
markersize = 4
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
metrics_to_plot = ['acc', 'precision', 'recall', 'f1']

model_name = [
    [F_T_small_mulitloss_1122_metrics, 'H-starNet'],
    [F_cnn_bn_small_1122_metrics, 'HARNet'],
    [T_att_small_1122_metrics, 'SSTA'],

    [S3R_metrics, 'S3R'],
    [MobileNetV3Small_ETF_metrics, 'MobileNetV3'],
    # [resnet18_metrics, 'resnet18'],
    # [resnet18_vector_group_metrics, 'resnet18-1D']
]

# 为不同模型指定不同的 marker（可选：配色与 markers.py 保持一致）
model_styles = {
    'HARNet':      dict(marker='o', color='#1f77b4'),  # 圆点
    'SSTA':        dict(marker='^', color='#ff7f0e'),  # 上三角
    'H-starNet':   dict(marker='s', color='#2ca02c'),  # 方块
    'S3R':         dict(marker='D', color='#d62728'),  # 菱形
    'MobileNetV3': dict(marker='v', color='#9467bd'),  # 下三角
    'resnet18':    dict(marker='X', color='#8c564b'),  # X（若启用）
    'resnet18-1D': dict(marker='P', color='#e377c2'),  # 五角星（若启用）
}

ylabel_map = {
    'acc': 'Accuracy',
    'precision': 'Precision',
    'recall': 'Recall',
    'f1': 'F1'
}

for metric in metrics_to_plot:
    fig = plt.figure(figsize=(3.4, 2.4))
    ax = fig.add_subplot(111)
    # 确保背景不透明，避免 EPS 透明度警告
    fig.patch.set_alpha(1.0)
    ax.patch.set_alpha(1.0)
    for metrics_dict, name in model_name:
        style = model_styles.get(name, {})
        ax.plot(
            SNR,
            metrics_dict[metric],
            '-',
            marker=style.get('marker', 'o'),
            color=style.get('color', None),
            linewidth=1.5,
            markersize=markersize,
            label=name,
        )
    ax.set_xlabel('SNR (dB)', fontsize=10, fontweight='bold')
    ax.set_ylabel(f"{ylabel_map.get(metric, metric)} (%)", fontsize=10, fontweight='bold')
    ax.grid(True, linewidth=0.6, alpha=0.4)
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.legend(loc='upper left', fontsize=7, framealpha=0.5, borderpad=0.3, handletextpad=0.4)
    fig.tight_layout()
    base = f'plot_graph/graph/{metric}'
    fig.savefig(base + '.png', dpi=300, transparent=False, bbox_inches='tight', pad_inches=0.02)
    # EPS: 去掉透明度 (transparent=False) 保证无警告；同时提供 PDF 作为备选
    fig.savefig(base + '.eps', format='eps', dpi=300, transparent=False, bbox_inches='tight', pad_inches=0.02)
    fig.savefig(base + '.pdf', format='pdf', dpi=300, transparent=False, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)