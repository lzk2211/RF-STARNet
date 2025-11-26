import matplotlib.pyplot as plt
from metrics_data3 import *

# 四类指标全部输出并保存 PNG + EPS
markersize = 4
metrics_to_plot = ['acc', 'precision', 'recall', 'f1']

model_name = [
    # [F_cnn_nano_1119_metrics, 'HARNet(nano)'],
    # [F_cnn_mini_1119_metrics, 'HARNet(mini)'],
    # [F_cnn_tiny_1119_metrics, 'HARNet(tiny)'],
    # [F_cnn_small_1119_metrics, 'HARNet(small)'],
    [F_cnn_base_1119_metrics, 'HARNet(base)'],
    # [F_cnn_large_1119_metrics, 'HARNet(large)'],
    # [T_att_nano_1119_metrics, 'SSTA(nano)'],
    # [T_att_mini_1119_metrics, 'SSTA(mini)'],
    # [T_att_tiny_1119_metrics, 'SSTA(tiny)'],
    # [T_att_small_1119_metrics, 'SSTA(small)'],
    [T_att_base_1119_metrics, 'SSTA(base)'],
    # [T_att_large_1119_metrics, 'SSTA(large)'],

    # [F_T_nano_gate_mulitloss_1119_metrics, 'HARNet+SSTA(nano)'],
    [F_T_mini_gate_mulitloss_1119_metrics, 'HARNet+SSTA(mini)'],
    [F_T_tiny_gate_mulitloss_1119_metrics, 'HARNet+SSTA(tiny)'],
    # [F_T_base_gate_mulitloss_1119_metrics, 'HARNet+SSTA(base)'],
    # [F_T_large_gate_mulitloss_1119_metrics, 'HARNet+SSTA(large)'],
    

    [S3R_metrics, 'S3R'],
    [MobileNetV3Small_ETF_metrics, 'MobileNetV3'],
    [resnet18_metrics, 'resnet18']
]

ylabel_map = {
    'acc': 'Accuracy',
    'precision': 'Precision',
    'recall': 'Recall',
    'f1': 'F1'
}

for metric in metrics_to_plot:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # 确保背景不透明，避免 EPS 透明度警告
    fig.patch.set_alpha(1.0)
    ax.patch.set_alpha(1.0)
    for metrics_dict, name in model_name:
        ax.plot(SNR, metrics_dict[metric], '-v', linewidth=1.5, markersize=markersize, label=name)
    ax.set_xlabel('SNR (dB)', fontsize=12)
    ax.set_ylabel(f"{ylabel_map.get(metric, metric)} (%)", fontsize=12)
    ax.grid(True)
    ax.tick_params(axis='both', which='major', labelsize=11)
    ax.legend(loc='upper left')
    fig.tight_layout()
    base = f'plot_graph/graph/{metric}'
    fig.savefig(base + '.png', dpi=300, transparent=False)
    # EPS: 去掉透明度 (transparent=False) 保证无警告；同时提供 PDF 作为备选
    fig.savefig(base + '.eps', format='eps', dpi=300, transparent=False)
    fig.savefig(base + '.pdf', format='pdf', dpi=300, transparent=False)
    plt.close(fig)