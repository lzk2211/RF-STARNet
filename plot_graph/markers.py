import matplotlib.pyplot as plt
import os
import numpy as np
from metrics_data4_5test import *

# 全局字体设为 Times New Roman（若系统未安装会回退）
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
def _plot_grouped_markers(groups_config, x_key, xlabel, title, out_name, x_log=True, figsize=(3.5, 2.6), font_cfg=None):
    """
    通用绘制函数：按给定 x_key (如 'flops' 或 'params') 作为横轴，Acc(%) 作为纵轴。
    每个组使用固定颜色和 marker，并按 x 排序后用同色线连接相邻点。
    """
    # 默认字体尺寸方案：适配双栏论文的单栏图
    if font_cfg is None:
        font_cfg = {"label": 10, "title": 11, "tick": 9, "legend": 7}

    out_dir = "plot_graph/graph"
    os.makedirs(out_dir, exist_ok=True)

    plt.figure(figsize=figsize)

    for group_name, cfg in groups_config.items():
        models = cfg["data"]
        color = cfg["color"]
        marker = cfg["marker"]

        if not models:
            continue

        # 取横轴与纵轴数据
        x_list = [m[1].get(x_key, None) for m in models]
        acc_list = [m[1]["avg_acc"] * 100 for m in models]

        # 过滤缺失值（以防个别模型无该字段）
        valid = [(x, acc, name) for (name, m), x, acc in zip(models, x_list, acc_list) if x is not None]
        if not valid:
            continue

        x_vals = np.array([v[0] for v in valid])
        y_vals = np.array([v[1] for v in valid])

        # 按 x 排序以整洁连线
        order = np.argsort(x_vals)
        x_sorted = x_vals[order]
        y_sorted = y_vals[order]

        # 散点
        plt.scatter(
            x_sorted,
            y_sorted,
            marker=marker,
            c=color,
            s=50,
            alpha=0.9,
            edgecolors="k",
            linewidths=0.4,
            label=group_name,
            zorder=3,
        )

        # 连线（单点组也允许，长度为零）
        plt.plot(
            x_sorted,
            y_sorted,
            color=color,
            linewidth=1.0,
            alpha=0.7,
            zorder=2,
        )

    if x_log:
        plt.xscale("log")
    plt.xlabel(xlabel, fontsize=font_cfg["label"], fontweight="bold", family="Times New Roman")
    plt.ylabel("Average Accuracy (%)", fontsize=font_cfg["label"], fontweight="bold", family="Times New Roman")
    # plt.title(title, fontsize=font_cfg["title"], pad=8, family="Times New Roman")

    plt.xticks(fontsize=font_cfg["tick"], family="Times New Roman")
    plt.yticks(fontsize=font_cfg["tick"], family="Times New Roman")

    plt.grid(True, which="major", ls="-", alpha=0.4)
    plt.grid(True, which="minor", ls=":", alpha=0.2)
    plt.legend(
        loc="lower right",
        ncol=3,
        fontsize=font_cfg["legend"],
        borderpad=0.4,
        framealpha=0.5,
        shadow=False,
        columnspacing=0.8,
        handletextpad=0.5,
    )
    plt.tight_layout()

    png_path = os.path.join(out_dir, f"{out_name}.png")
    eps_path = os.path.join(out_dir, f"{out_name}.eps")
    plt.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.savefig(eps_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.show()

def create_markers_scatter_chart():
    """
    旧接口：分别输出 FLOPs/Params 两张单图（保留以兼容）。
    """
    # 三个主类别与各版本数据
    group_fcnn = [
        ("Nano", F_cnn_bn_nano_1122_metrics),
        ("Mini", F_cnn_bn_mini_1122_metrics),
        ("Tiny", F_cnn_bn_tiny_1122_metrics),
        ("Small", F_cnn_bn_small_1122_metrics),
    ]

    group_tatt = [
        ("Nano", T_att_nano_1122_metrics),
        ("Mini", T_att_mini_1122_metrics),
        ("Tiny", T_att_tiny_1122_metrics),
        ("Small", T_att_small_1122_metrics),
    ]

    group_fusion = [
        ("Nano", F_T_nano_mulitloss_1122_metrics),
        ("Mini", F_T_mini_mulitloss_1122_metrics),
        ("Tiny", F_T_tiny_mulitloss_1122_metrics),
        ("Small", F_T_small_mulitloss_1122_metrics),
    ]

    # 其它对比模型
    group_s3r = [("S3R", S3R_metrics)]
    group_mobilenet = [("MobileNetV3Small", MobileNetV3Small_ETF_metrics)]
    group_resnet = [
        ("ResNet18", resnet18_metrics)
    ]
    group_1d_resnet = [
        ("ResNet18-VecGroup", resnet18_vector_group_metrics)
    ]

    groups_config = {
        "SARNet": {"data": group_fcnn, "color": "#1f77b4", "marker": "o"},          # 蓝色 圆点
        "SSTA": {"data": group_tatt, "color": "#ff7f0e", "marker": "^"},          # 橙色 三角
        "RF-starNet":  {"data": group_fusion, "color": "#2ca02c", "marker": "s"},        # 绿色 方块
        "S3R":  {"data": group_s3r, "color": "#d62728", "marker": "D"},           # 红色 菱形
        "MobileNet": {"data": group_mobilenet, "color": "#9467bd", "marker": "v"},# 紫色 倒三角
        "ResNet18": {"data": group_resnet, "color": "#8c564b", "marker": "X"},      # 棕色 X
        "ResNet18-1D": {"data": group_1d_resnet, "color": "#e377c2", "marker": "P"},# 粉色 五角星
    }

    # 图1：FLOPs（计算成本）
    _plot_grouped_markers(
        groups_config,
        x_key="flops",
        xlabel="Model MACs (FLOPs)",
        title="Accuracy vs. Efficiency (FLOPs)",
        out_name="markers_chart_flops",
        x_log=True,
    )

    # 图2：Params（参数量）
    _plot_grouped_markers(
        groups_config,
        x_key="params",
        xlabel="Model Params (M)",
        title="Accuracy vs. Efficiency (Params)",
        out_name="markers_chart_params",
        x_log=True,
    )


if __name__ == "__main__":
    create_markers_scatter_chart()
