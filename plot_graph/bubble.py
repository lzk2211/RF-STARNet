import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from metrics_data3 import *

# 全局字体设为 Times New Roman（若系统未安装会回退）
plt.rcParams['font.serif']
# ==========================================
# 2. Helper Functions (Contrast Adjusted)
# ==========================================

def calculate_log_size(params):
    """
    计算 Log 缩放后的大小 (高对比度版)。
    逻辑：(log10(params) - offset) * scale
    """
    log_val = np.log10(params)
    
    # log_val 范围: ~3.7 (Nano) 到 ~7.5 (S3R)
    # 调整 offset 到 3.5 (接近最小值的 log)，使小模型非常小
    # 调整 scale 到 400，使差异倍数被放大
    # Nano: (3.74 - 3.5)*400 = ~96
    # Large: (5.6 - 3.5)*400 = ~840 (约9倍差异)
    # S3R: (7.5 - 3.5)*400 = ~1600
    size = (log_val - 3.5) * 400 
    return max(size, 50) # 保证最小值可见


def create_high_contrast_bubble_chart():
    # 3. Grouping Data
    group_a = [
        ("Nano", F_cnn_nano_1119_metrics), 
        ("Mini", F_cnn_mini_1119_metrics),
        ("Tiny", F_cnn_tiny_1119_metrics), 
        ("Small", F_cnn_small_1119_metrics),
        ("Base", F_cnn_base_1119_metrics), 
        ("Large", F_cnn_large_1119_metrics)
    ]
    
    group_b = [
        ("Nano", T_att_nano_1119_metrics), 
        ("Mini", T_att_mini_1119_metrics),
        ("Tiny", T_att_tiny_1119_metrics), 
        ("Small", T_att_small_1119_metrics),
        ("Base", T_att_base_1119_metrics), 
        ("Large", T_att_large_1119_metrics)
    ]
    
    group_fusion = [
        ("Nano", F_T_nano_gate_mulitloss_1119_metrics), 
        ("Mini", F_T_mini_gate_mulitloss_1119_metrics),
        ("Tiny", F_T_tiny_gate_mulitloss_1119_metrics), 
        ("Small", F_T_small_gate_mulitloss_1119_metrics),
        ("Base", F_T_base_gate_mulitloss_1119_metrics), 
        ("Large", F_T_large_gate_mulitloss_1119_metrics)
    ]
    
    group_sota = [
        ("MobileNetV3", MobileNetV3Small_ETF_metrics),
        ("ResNet18", resnet18_metrics),
        ("S3R", S3R_metrics)
    ]

    groups_config = {
        "HARNet (Branch A)": {"data": group_a, "cmap": plt.cm.Blues, "base_color": "#1f77b4"},
        "SSTA (Branch B)":   {"data": group_b, "cmap": plt.cm.Oranges, "base_color": "#ff7f0e"},
        "Fusion (A+B)":      {"data": group_fusion, "cmap": plt.cm.Greens, "base_color": "#2ca02c"},
        "Baseline (SOTA)":   {"data": group_sota, "cmap": plt.cm.Purples, "base_color": "#9467bd"}
    }

    # 4. Plotting
    plt.figure(figsize=(14, 10))
    legend_handles = []
    
    for group_name, config in groups_config.items():
        models = config["data"]
        cmap = config["cmap"]
        
        flops_list = [m[1]['flops'] for m in models]
        acc_list = [m[1]['avg_acc'] * 100 for m in models] # Convert to %
        params_list = [m[1]['params'] for m in models]
        names = [m[0] for m in models]
        
        num_models = len(models)
        
        # === 修改点 1: 颜色渐变拉大 ===
        # 使用 0.3 到 1.0 的范围 (之前是 0.4 到 1.0)
        # 0.3 更浅，1.0 最深，对比更明显
        if num_models > 1:
            color_indices = np.linspace(0.3, 1.0, num_models)
            colors = cmap(color_indices)
        else:
            colors = cmap([0.7]) 

        # === 修改点 2: 圆圈大小计算 ===
        sizes = [calculate_log_size(p) for p in params_list]
        
        # Scatter Plot
        plt.scatter(flops_list, acc_list, s=sizes, c=colors,
                label=group_name, alpha=0.85)
        
        # Legend Handle
        legend_handles.append(mpatches.Patch(color=config["base_color"], label=group_name))

        # Labels
        for i, name in enumerate(names):
            x = flops_list[i]
            y = acc_list[i]
            
            # 微调标签位置防止遮挡
            y_offset = 0.6
            va = 'bottom'
            
            if name == "S3R": 
                y_offset = -1.8
                va = 'top'
            elif name == "ResNet18":
                y_offset = -1.8
                va = 'top'
            elif name == "MobileNetV3":
                y_offset = 0.9
            elif name == "Base" and group_name == "Fusion (A+B)":
                y_offset = -1.2
                va = 'top'
            
            plt.text(x, y + y_offset, name, fontsize=9, ha='center', va=va, 
                     fontweight='bold', color='black', alpha=0.9)

    # 5. Axis & Styling
    plt.xscale('log')
    plt.xlabel('Computational Cost (FLOPs)', fontsize=14, fontweight='bold')
    plt.ylabel('Average Accuracy (%)', fontsize=14, fontweight='bold')
    plt.title('Accuracy vs. Efficiency (Higher Contrast)', fontsize=16, pad=20)
    
    plt.grid(True, which="major", ls="-", alpha=0.4)
    plt.grid(True, which="minor", ls=":", alpha=0.2)

    # 6. Legend (Log Scale Examples)
    size_examples = [10000, 100000, 1000000, 10000000] 
    size_labels = ["10k", "100k", "1M", "10M"]
    
    legend_handles.append(mpatches.Patch(color='none', label=' '))
    legend_handles.append(mpatches.Patch(color='none', label=r'$\bf{Params\ (Log\ Scale)}$'))
    
    for size_val, label_text in zip(size_examples, size_labels):
        s = calculate_log_size(size_val)
        h = plt.scatter([], [], s=s, c='silver', alpha=0.5, label=label_text)
        legend_handles.append(h)

    plt.legend(handles=legend_handles, loc='lower right', fontsize=10, 
               borderpad=1.2, framealpha=0.9, shadow=True)

    plt.tight_layout()
    plt.savefig('plot_graph/graph/bubble_chart_high_contrast.png', dpi=300)
    plt.savefig('plot_graph/graph/bubble_chart_high_contrast.eps', dpi=300)
    plt.show()

if __name__ == "__main__":
    create_high_contrast_bubble_chart()