import matplotlib.pyplot as plt
from metrics_data4_5test import *

markersize = 4
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

metrics_to_plot = ['acc', 'precision', 'recall', 'f1']

model_name = [
    [F_T_small_mulitloss_1122_metrics, 'RF-STARNet'],
    [F_cnn_bn_small_1122_metrics, 'SARNet'],
    [T_att_small_1122_metrics, 'SSTA'],
    [S3R_metrics, 'S3R'],
    [MobileNetV3Small_ETF_metrics, 'MobileNetV3'],
    # [resnet18_metrics, 'resnet18'],
    # [resnet18_vector_group_metrics, 'resnet18-1D']
]

model_styles = {
    'SARNet':      dict(marker='o', color='#1f77b4'),
    'SSTA':        dict(marker='^', color='#ff7f0e'),
    'RF-STARNet':   dict(marker='s', color='#2ca02c'),
    'S3R':         dict(marker='D', color='#d62728'),
    'MobileNetV3': dict(marker='v', color='#9467bd'),
    'resnet18':    dict(marker='X', color='#8c564b'),
    'resnet18-1D': dict(marker='P', color='#e377c2'),
}

ylabel_map = {
    'acc': 'Accuracy',
    'precision': 'Precision',
    'recall': 'Recall',
    'f1': 'F1-score'
}

fig, axes = plt.subplots(1, 4, figsize=(9.5, 2.5), constrained_layout=True)

handles = []
labels = []

for i, metric in enumerate(metrics_to_plot):
    ax = axes[i]
    
    for metrics_dict, name in model_name:
        style = model_styles.get(name, {})
        line, = ax.plot(
            SNR,
            metrics_dict[metric],
            '-',
            marker=style.get('marker', 'o'),
            color=style.get('color', None),
            linewidth=1.5,
            markersize=markersize,
            label=name,
        )
        if i == 0:
            handles.append(line)
            labels.append(name)

    sub_title = f"({chr(97+i)}) {ylabel_map.get(metric, metric)}"
    ax.set_xlabel(f"SNR (dB)\n{sub_title}", fontsize=12)
    
    ax.grid(True, linewidth=0.6, alpha=0.4)
    ax.tick_params(axis='both', which='major', labelsize=10)
    
    if i == 0:
        ax.set_ylabel("Value (%)", fontsize=10)

fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.15), 
           ncol=len(model_name), fontsize=10, frameon=False)
base = './plot_graph/graph/combined_metrics_plot'
fig.savefig(base + '.png', dpi=300, bbox_inches='tight')
fig.savefig(base + '.eps', format='eps', dpi=300, transparent=False, bbox_inches='tight')
fig.savefig(base + '.pdf', format='pdf', dpi=300, transparent=False, bbox_inches='tight')
