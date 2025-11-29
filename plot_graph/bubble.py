import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from metrics_data3 import *

# 全局字体设为 Times New Roman（若系统未安装会回退）
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
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
        # ("Base", F_cnn_base_1119_metrics), 
        # ("Large", F_cnn_large_1119_metrics)
    ]
    
    group_b = [
        ("Nano", T_att_nano_1119_metrics), 
        ("Mini", T_att_mini_1119_metrics),
        ("Tiny", T_att_tiny_1119_metrics), 
        ("Small", T_att_small_1119_metrics),
        # ("Base", T_att_base_1119_metrics), 
        # ("Large", T_att_large_1119_metrics)
    ]
    
    group_fusion = [
        ("Nano", F_T_nano_gate_mulitloss_1119_metrics), 
        ("Mini", F_T_mini_gate_mulitloss_1119_metrics),
        ("Tiny", F_T_tiny_gate_mulitloss_1119_metrics), 
        ("Small", F_T_small_gate_mulitloss_1119_metrics),
        # ("Base", F_T_base_gate_mulitloss_1119_metrics), 
        # ("Large", F_T_large_gate_mulitloss_1119_metrics)
    ]
    
    group_sota = [
        ("MobileNetV3", MobileNetV3Small_ETF_metrics),
        ("ResNet18", resnet18_metrics),
        ("S3R", S3R_metrics),
        ("1D-LW-ResNet18", resnet18_vector_group_metrics),
    ]

    groups_config = {
        "HARNet": {"data": group_a, "cmap": plt.cm.Blues, "base_color": "#1f77b4"},
        "SSTA":   {"data": group_b, "cmap": plt.cm.Oranges, "base_color": "#ff7f0e"},
        "H-starNet":      {"data": group_fusion, "cmap": plt.cm.Greens, "base_color": "#2ca02c"},
        "Baseline":          {"data": group_sota, "cmap": plt.cm.Purples, "base_color": "#9467bd"}
    }

    # 4. Plotting
    # 图尺寸：论文中将缩到约 8.9mm 宽时仍需清晰，
    # 这里使用较紧凑的宽度且字号加大，以便缩放后可读。
    plt.figure(figsize=(8, 6))
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
        if group_name == "Baseline":
            # 基线集合：按参数规模映射颜色深度（参数越大越深）
            params_arr = np.array(params_list, dtype=float)
            if params_arr.size == 0:
                color_indices = np.array([0.7])
            else:
                rng = params_arr.max() - params_arr.min()
                if rng < 1e-12:
                    norm = np.zeros_like(params_arr)
                else:
                    norm = (params_arr - params_arr.min()) / rng  # 0..1
                color_indices = 0.3 + 0.7 * norm  # 0.3(浅) -> 1.0(深)
            colors = cmap(color_indices)
        else:
            if num_models > 1:
                color_indices = np.linspace(0.3, 1.0, num_models)
                colors = cmap(color_indices)
            else:
                colors = cmap([0.7]) 

        # === 修改点 2: 圆圈大小计算 ===
        sizes = [calculate_log_size(p) for p in params_list]
        # 对连线组强制尺寸严格递增（按给定顺序 Nano->Mini->Tiny->Small）
        if group_name != "Baseline" and len(sizes) > 1:
            # 线性递增映射，保持总体尺度范围但确保严格单调
            s_min = min(sizes)
            s_max = max(sizes)
            # 若所有原始大小几乎相等，人工扩展范围
            if abs(s_max - s_min) < 1e-6:
                s_min = s_min * 0.8
                s_max = s_max * 1.2
            new_sizes = np.linspace(s_min, s_max, len(sizes))
            # 保证严格递增（linspace 已满足），覆盖原尺寸
            sizes = list(new_sizes)
        
        # Scatter Plot
        # 按 FLOPs 从小到大排序，便于连线连接
        order = np.argsort(flops_list)
        flops_sorted = np.array(flops_list)[order]
        acc_sorted = np.array(acc_list)[order]
        sizes_sorted = np.array(sizes)[order]
        names_sorted = np.array(names)[order]
        colors_sorted = np.array(colors)[order]

        # 根据排序后的索引同步新尺寸（已保证递增）
        sizes_sorted = np.array(sizes)[order]
        # 散点
        plt.scatter(flops_sorted, acc_sorted, s=sizes_sorted, c=colors_sorted,
                label=group_name, alpha=0.85, edgecolors='k', linewidths=0.3)

        # 用折线连接该组圆圈中心（Baseline 不连线）
        if group_name != "Baseline":
            plt.plot(flops_sorted, acc_sorted, color=config["base_color"], linewidth=1.5, alpha=0.8)
        
        # Legend Handle
        legend_handles.append(mpatches.Patch(color=config["base_color"], label=group_name))

        # 点标签：仅为 Baseline 保留，其它连线组去除标签
        if group_name == "Baseline":
            for i, name in enumerate(names):
                x = flops_list[i]
                y = acc_list[i]
                # 根据参数量向上或向下偏移，减少重叠
                y_offset = 0.6
                va = 'bottom'
                if "ResNet" in name:
                    y_offset = -1.6
                    va = 'top'
                elif "S3R" in name:
                    y_offset = -2.0
                    va = 'top'
                elif "1D-LW" in name:
                    y_offset = 0.9
                    va = 'bottom'
                plt.text(x, y + y_offset, name, fontsize=12, ha='center', va=va,
                         fontweight='bold', color='black', alpha=0.9, family='Times New Roman')

    # 5. Axis & Styling
    plt.xscale('log')
    plt.xlabel('Computational Cost (FLOPs)', fontsize=16, fontweight='bold', family='Times New Roman')
    plt.ylabel('Average Accuracy (%)', fontsize=16, fontweight='bold', family='Times New Roman')
    plt.title('Accuracy vs. Efficiency (Higher Contrast)', fontsize=18, pad=16, family='Times New Roman')
    
    plt.grid(True, which="major", ls="-", alpha=0.4)
    plt.grid(True, which="minor", ls=":", alpha=0.2)

    # 6. Legend (simplified)
    plt.legend(handles=legend_handles, loc='lower right', fontsize=11,
               borderpad=0.8, framealpha=0.5, shadow=False)

    plt.tight_layout()
    plt.savefig('plot_graph/graph/bubble_chart_high_contrast.png', dpi=300)
    plt.savefig('plot_graph/graph/bubble_chart_high_contrast.eps', dpi=300)
    plt.show()

if __name__ == "__main__":
    create_high_contrast_bubble_chart()