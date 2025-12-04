import os
import numpy as np

__all__ = [
    'init_branch_results',
    'update_branch_results',
    'plot_branch_venn'
]

def init_branch_results():
    """初始化统计字典，覆盖用户所需七类统计。"""
    return {
        # 基础两分支统计（保留原键，便于兼容）
        'both_correct': 0,
        'fcnn_only': 0,
        'tatt_only': 0,
        'both_wrong': 0,
        'fusion_rescued': 0,
        # 用户明确需要展示的七项
        'only_fcnn': 0,                      # 只有 fcnn 对
        'only_tatt': 0,                      # 只有 tatt 对
        'both_fcnn_tatt': 0,                 # fcnn 和 tatt 都对
        'only_fusion': 0,                    # 只有 fusion 对
        'fusion_with_branch_correct': 0,     # fusion 对，且 fcnn 或 tatt 至少一个也对
        'fusion_only': 0,                    # fusion 对，但 fcnn 和 tatt 都错
        'branch_correct_but_fusion_wrong': 0 # fcnn 或 tatt 对了，但 fusion 错
    }

def update_branch_results(results, preds_res, preds_sp, preds_fusion, labels):
    """基于一个 batch 的预测更新统计字典，满足用户的七项展示需求。"""
    crt_res = (preds_res == labels).cpu().numpy().astype(np.int32)
    crt_sp = (preds_sp == labels).cpu().numpy().astype(np.int32)
    crt_fusion = (preds_fusion == labels).cpu().numpy().astype(np.int32)

    # 保留原始四象限 + rescue 统计（兼容）
    both_correct = np.sum((crt_res == 1) & (crt_sp == 1))
    fcnn_only = np.sum((crt_res == 1) & (crt_sp == 0))
    tatt_only = np.sum((crt_res == 0) & (crt_sp == 1))
    both_wrong = np.sum((crt_res == 0) & (crt_sp == 0))
    fusion_rescued = np.sum(((crt_res ^ crt_sp) == 1) & (crt_fusion == 1))

    results['both_correct'] += int(both_correct)
    results['fcnn_only'] += int(fcnn_only)
    results['tatt_only'] += int(tatt_only)
    results['both_wrong'] += int(both_wrong)
    results['fusion_rescued'] += int(fusion_rescued)

    # 用户七项统计
    only_fcnn = np.sum((crt_res == 1) & (crt_sp == 0) & (crt_fusion == 0))
    only_tatt = np.sum((crt_res == 0) & (crt_sp == 1) & (crt_fusion == 0))
    both_fcnn_tatt = np.sum((crt_res == 1) & (crt_sp == 1))
    only_fusion = np.sum((crt_fusion == 1) & (crt_res == 0) & (crt_sp == 0))
    fusion_with_branch_correct = np.sum((crt_fusion == 1) & ((crt_res == 1) | (crt_sp == 1)))
    fusion_only = np.sum((crt_fusion == 1) & (crt_res == 0) & (crt_sp == 0))
    branch_correct_but_fusion_wrong = np.sum(((crt_res == 1) | (crt_sp == 1)) & (crt_fusion == 0))

    results['only_fcnn'] += int(only_fcnn)
    results['only_tatt'] += int(only_tatt)
    results['both_fcnn_tatt'] += int(both_fcnn_tatt)
    results['only_fusion'] += int(only_fusion)
    results['fusion_with_branch_correct'] += int(fusion_with_branch_correct)
    results['fusion_only'] += int(fusion_only)
    results['branch_correct_but_fusion_wrong'] += int(branch_correct_but_fusion_wrong)

    return results

def plot_branch_venn(results, out_dir):
    """绘制三集合维恩图 (FCNN / T-ATT / Fusion)，并在标题中展示用户关注的计数。"""
    try:
        import matplotlib.pyplot as plt
        try:
            from matplotlib_venn import venn3
            fig, ax = plt.subplots(figsize=(6, 5))
            v = venn3(subsets=(
                results.get('only_fcnn', 0),
                results.get('only_tatt', 0),
                results.get('both_fcnn_tatt', 0),
                results.get('only_fusion', 0),
                results.get('fusion_with_branch_correct', 0) - results.get('both_fcnn_tatt', 0),  # FCNN∩Fusion 但不含 TATT
                results.get('fusion_with_branch_correct', 0) - results.get('both_fcnn_tatt', 0),  # TATT∩Fusion 但不含 FCNN (近似展示)
                results.get('all_three', 0) if 'all_three' in results else results.get('both_fcnn_tatt', 0)  # 若未统计则以双对近似
            ), set_labels=('FCNN', 'T-ATT', 'Fusion'))
            ax.set_title('FCNN / T-ATT / Fusion\n'
                         f"only_fcnn={results.get('only_fcnn',0)}, only_tatt={results.get('only_tatt',0)}, both_fcnn_tatt={results.get('both_fcnn_tatt',0)}\n"
                         f"only_fusion={results.get('only_fusion',0)}, fusion_with_branch_correct={results.get('fusion_with_branch_correct',0)}, fusion_only={results.get('fusion_only',0)}\n"
                         f"branch_correct_but_fusion_wrong={results.get('branch_correct_but_fusion_wrong',0)}")
        except Exception:
            from matplotlib.patches import Circle
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.set_aspect('equal')
            circles = [
                Circle((0.40, 0.55), 0.28, color='tab:blue', alpha=0.30),
                Circle((0.60, 0.55), 0.28, color='tab:orange', alpha=0.30),
                Circle((0.50, 0.35), 0.28, color='tab:green', alpha=0.30)
            ]
            for c in circles: ax.add_patch(c)
            ax.text(0.30, 0.55, str(results.get('only_fcnn',0)), ha='center')
            ax.text(0.70, 0.55, str(results.get('only_tatt',0)), ha='center')
            ax.text(0.50, 0.20, str(results.get('only_fusion',0)), ha='center')
            ax.text(0.50, 0.60, str(results.get('both_fcnn_tatt',0)), ha='center', fontweight='bold')
            ax.text(0.50, 0.05, f"fusion_with_branch_correct={results.get('fusion_with_branch_correct',0)}\n"
                                f"fusion_only={results.get('fusion_only',0)}\n"
                                f"branch_correct_but_fusion_wrong={results.get('branch_correct_but_fusion_wrong',0)}",
                    ha='center', va='center')
            ax.text(0.40, 0.85, 'FCNN', ha='center')
            ax.text(0.60, 0.85, 'T-ATT', ha='center')
            ax.text(0.50, 0.02, 'Fusion', ha='center')
            ax.set_xlim(0,1); ax.set_ylim(0,0.95); ax.axis('off')
        out_png = os.path.join(out_dir, 'venn_fcnn_tatt_fusion.png')
        out_pdf = os.path.join(out_dir, 'venn_fcnn_tatt_fusion.pdf')
        fig.savefig(out_png, dpi=300, bbox_inches='tight')
        fig.savefig(out_pdf, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved Venn diagram to: {out_png}")
    except Exception as e:
        print(f"[Warn] Failed to plot Venn diagram: {e}")
