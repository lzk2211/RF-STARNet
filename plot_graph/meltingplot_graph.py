import matplotlib.pyplot as plt
from meltingdata import results, SNR

"""
改进版绘图：
1. 使用 Times New Roman 字体，适合论文排版。
2. 四个指标 (Accuracy / Precision / Recall / F1) 同时在 2x2 子图中展示，方便比较。
3. 仍保留单独每指标一张图的导出，便于灵活使用。
4. 模型列表改为 list[ (name, metrics_dict) ] 结构，避免原交替结构容易误用。
5. 自动检查长度匹配，避免数组长度不一致导致崩溃。
6. 统一配色与标记，增强区分度；支持百分比刻度 (自动乘以 100 如果原始值是 0~1)。
7. 图例集中放置在总图右下，透明背景。
"""

# 全局样式
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['axes.unicode_minus'] = False

metrics_to_plot = ['acc', 'precision', 'recall', 'f1']
ylabel_map = {
    'acc': 'Accuracy (%)',
    'precision': 'Precision (%)',
    'recall': 'Recall (%)',
    'f1': 'F1 (%)'
}

# 模型集合（名称, metrics_dict）
model_entries = [
    ('ResNet Tiny (Basic)',        results['42']['F_cnn_basic_tiny_1122_metrics']),
    ('ResNet Tiny (SE)',           results['42']['F_cnn_SE_tiny_1122_metrics']),
    ('ResNet Tiny (MultiDilation)',results['42']['F_cnn_MultiDilation_tiny_1122_metrics']),
    ('HARNet Tiny',                results['42']['F_cnn_bn_tiny_1122_metrics']),
]

# 颜色与标记循环
color_cycle = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
marker_cycle = ['o', 's', 'D', '^', 'v', 'P']

def _to_percent(arr):
    arr = list(arr)
    if not arr:
        return arr
    # 若最大值 <= 1.2 认为是 0~1 之间的比例，乘 100
    if max(arr) <= 1.2:
        return [x * 100.0 for x in arr]
    return arr

def _validate_lengths(snr, series):
    return len(snr) == len(series)

def plot_combined():
    fig, axes = plt.subplots(2, 2, figsize=(8, 6))
    axes_flat = axes.flatten()

    # 绘制每个指标
    for idx, metric in enumerate(metrics_to_plot):
        ax = axes_flat[idx]
        for mi, (name, mdict) in enumerate(model_entries):
            if metric not in mdict:
                continue
            series = _to_percent(mdict[metric])
            if not _validate_lengths(SNR, series):
                continue
            ax.plot(
                SNR,
                series,
                marker=marker_cycle[mi % len(marker_cycle)],
                color=color_cycle[mi % len(color_cycle)],
                linewidth=1.6,
                markersize=5,
                alpha=0.9,
                label=name if idx == 0 else None  # 只在第一个子图收集图例
            )
        ax.set_xlabel('SNR (dB)', fontsize=12)
        ax.set_ylabel(ylabel_map.get(metric, metric), fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.35)
        ax.tick_params(axis='both', labelsize=10)
        # y 轴范围尝试自动设置
        ax.set_ylim(0, 105)

    # 统一图例（从第一个子图获取）
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower right', fontsize=10, framealpha=0.5)
    fig.suptitle('Performance vs. SNR', fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    base = 'plot_graph/graph/combined_metrics'
    fig.savefig(base + '.png')
    fig.savefig(base + '.pdf')
    fig.savefig(base + '.eps', format='eps')
    plt.close(fig)

def plot_individual():
    for metric in metrics_to_plot:
        fig, ax = plt.subplots(figsize=(4, 3.2))
        for mi, (name, mdict) in enumerate(model_entries):
            if metric not in mdict:
                continue
            series = _to_percent(mdict[metric])
            if not _validate_lengths(SNR, series):
                continue
            ax.plot(
                SNR,
                series,
                marker=marker_cycle[mi % len(marker_cycle)],
                color=color_cycle[mi % len(color_cycle)],
                linewidth=1.6,
                markersize=5,
                alpha=0.9,
                label=name
            )
        ax.set_xlabel('SNR (dB)', fontsize=12)
        ax.set_ylabel(ylabel_map.get(metric, metric), fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.35)
        ax.tick_params(axis='both', labelsize=10)
        ax.set_ylim(0, 105)
        ax.legend(loc='lower right', fontsize=9, framealpha=0.5)
        fig.tight_layout()
        base = f'plot_graph/graph/{metric}'
        fig.savefig(base + '.png')
        fig.savefig(base + '.pdf')
        fig.savefig(base + '.eps', format='eps')
        plt.close(fig)

if __name__ == '__main__':
    plot_combined()
    plot_individual()