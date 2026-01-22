import matplotlib.pyplot as plt
import os
import numpy as np
from metrics_data4_5test import *

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

def _plot_grouped_markers(groups_config, x_key, xlabel, title, out_name, x_log=True, figsize=(3.5, 2.6), font_cfg=None):
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

        x_list = [m[1].get(x_key, None) for m in models]
        acc_list = [m[1]["avg_acc"] * 100 for m in models]

        valid = [(x, acc, name) for (name, m), x, acc in zip(models, x_list, acc_list) if x is not None]
        if not valid:
            continue

        x_vals = np.array([v[0] for v in valid])
        y_vals = np.array([v[1] for v in valid])

        order = np.argsort(x_vals)
        x_sorted = x_vals[order]
        y_sorted = y_vals[order]

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

    plt.xticks(fontsize=font_cfg["tick"], family="Times New Roman")
    plt.yticks(fontsize=font_cfg["tick"], family="Times New Roman")

    plt.grid(True, which="major", ls="-", alpha=0.4)
    plt.grid(True, which="minor", ls=":", alpha=0.2)
    plt.legend(
        loc="lower right",
        ncol=2,
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

    group_s3r = [("S3R", S3R_metrics)]
    group_mobilenet = [("MobileNetV3Small", MobileNetV3Small_ETF_metrics)]
    group_resnet = [
        ("ResNet18", resnet18_metrics)
    ]
    group_1d_resnet = [
        ("ResNet18-VecGroup", resnet18_vector_group_metrics)
    ]

    groups_config = {
        "SARNet": {"data": group_fcnn, "color": "#1f77b4", "marker": "o"},
        "SSTA": {"data": group_tatt, "color": "#ff7f0e", "marker": "^"},
        "RF-STARNet":  {"data": group_fusion, "color": "#2ca02c", "marker": "s"},
        "S3R":  {"data": group_s3r, "color": "#d62728", "marker": "D"},
        "MobileNetV3": {"data": group_mobilenet, "color": "#9467bd", "marker": "v"},
        "ResNet18": {"data": group_resnet, "color": "#8c564b", "marker": "X"},
        "ResNet18-1D": {"data": group_1d_resnet, "color": "#e377c2", "marker": "P"},
    }

    # Fig.1：FLOPs
    _plot_grouped_markers(
        groups_config,
        x_key="flops",
        xlabel="Model MACs (FLOPs)",
        title="Accuracy vs. Efficiency (FLOPs)",
        out_name="markers_chart_flops",
        x_log=True,
    )

    # Fig.2：Params
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
