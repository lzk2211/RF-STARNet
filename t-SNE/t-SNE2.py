import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from sklearn.manifold import TSNE
import seaborn as sns

# 复用你的模块
try:
    from fun import folder_settings, switch_model
    from dataset.ZJU_dataset_pre_multiSNR import data_set_split
except ImportError:
    pass

sns.set(style="whitegrid")

class ModelProbe:
    """
    全能探针：同时提取 (Res特征, Sp特征, Fusion特征) 和 (Res预测, Sp预测, Fusion预测)
    """
    def __init__(self, model):
        self.model = model
        self.data = {
            'feat_res': [], 'feat_sp': [], 'feat_fusion': [],
            'logits_res': [], 'logits_sp': [], 'logits_fusion': []
        }
        self.hooks = []
        
        # 注册 Hook
        # 1. 获取特征 (Linear层输入)
        self.hooks.append(model.aux_head_res.register_forward_hook(self._save_feat('feat_res')))
        self.hooks.append(model.aux_head_sp.register_forward_hook(self._save_feat('feat_sp')))
        self.hooks.append(model.classifier.register_forward_hook(self._save_feat('feat_fusion')))
        
        # 2. 获取 Logits (Linear层输出)
        self.hooks.append(model.aux_head_res.register_forward_hook(self._save_logits('logits_res')))
        self.hooks.append(model.aux_head_sp.register_forward_hook(self._save_logits('logits_sp')))
        self.hooks.append(model.classifier.register_forward_hook(self._save_logits('logits_fusion')))

    def _save_feat(self, name):
        def hook(module, input, output):
            # Input[0] 是特征
            f = input[0].flatten(1).detach().cpu()
            self.data[name].append(f)
        return hook

    def _save_logits(self, name):
        def hook(module, input, output):
            # Output 是 Logits
            l = output.detach().cpu()
            self.data[name].append(l)
        return hook

    def close(self):
        for h in self.hooks: h.remove()

    def get_results(self):
        # 拼接所有 batch
        final_res = {}
        for k, v in self.data.items():
            if v:
                final_res[k] = torch.cat(v, dim=0).numpy()
            else:
                final_res[k] = None
        return final_res

def run_analysis(args):
    # 1. Setup
    device = args.device
    print("Loading Data & Model...")
    _, _, test_loader, _ = data_set_split(0, args)
    model = switch_model(args.model, device, args.class_num)
    
    # Load weights
    model_path = os.path.join('./1111', args.exp_name, 'model.pth')
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print("[WARN] No weights found, using random!")

    model.to(device)
    model.eval()

    # 2. Inference with Probe
    probe = ModelProbe(model)
    all_labels = []
    
    print(f"Running Inference (Max {args.max_samples} samples)...")
    count = 0
    with torch.no_grad():
        for imgs, labels, snrs in test_loader:
            if count >= args.max_samples: break
            imgs = imgs.to(device)
            _ = model(imgs) # Hook 自动收集数据
            
            if isinstance(labels, torch.Tensor):
                all_labels.extend(labels.cpu().numpy())
            else:
                all_labels.extend(labels)
            
            count += imgs.size(0)
            print(f"Processed {count}...", end='\r')
    
    probe.close()
    data = probe.get_results()
    labels = np.array(all_labels)
    
    # 3. Analyze Predictions
    # 获取预测类别
    pred_res = np.argmax(data['logits_res'], axis=1)
    pred_sp = np.argmax(data['logits_sp'], axis=1)
    pred_fusion = np.argmax(data['logits_fusion'], axis=1)
    
    # --- 核心逻辑：寻找 A 拯救 B 的样本 ---
    # 条件：Sp 错了 AND Fusion 对了
    saved_mask = (pred_sp != labels) & (pred_fusion == labels)
    saved_indices = np.where(saved_mask)[0]
    
    print(f"\n\n[Analysis Result]")
    print(f"Total Samples: {len(labels)}")
    print(f"Sp Acc: {np.mean(pred_sp == labels):.2%}")
    print(f"Fusion Acc: {np.mean(pred_fusion == labels):.2%}")
    print(f"Samples 'Saved' by Fusion (Sp Wrong -> Fusion Correct): {len(saved_indices)}")
    
    if len(saved_indices) < 5:
        print("[WARN] 拯救样本太少，图可能不好看。建议增大 max_samples 或换个 SNR。")
    
    # 4. Visualization
    # 我们画两张图：
    # 左图：B 分支特征 (feat_sp)
    # 右图：Fusion 分支特征 (feat_fusion)
    # 策略：画所有点作为灰色背景，高亮 'Saved' 样本
    
    # 为了 t-SNE 稳定，我们用全量数据跑 t-SNE
    print("Running t-SNE on Sp features...")
    tsne_sp = TSNE(n_components=2, init='pca', learning_rate='auto', random_state=42).fit_transform(data['feat_sp'])
    
    print("Running t-SNE on Fusion features...")
    tsne_fusion = TSNE(n_components=2, init='pca', learning_rate='auto', random_state=42).fit_transform(data['feat_fusion'])

    # 绘图
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    unique_labels = np.unique(labels)
    palette = sns.color_palette("bright", len(unique_labels))
    
    # 辅助函数：画背景和前景
    def plot_on(ax, tsne_data, title):
        # 1. 背景：所有点 (灰色，透明)
        ax.scatter(tsne_data[:, 0], tsne_data[:, 1], c='lightgrey', s=10, alpha=0.3, label='Other Samples')
        
        # 2. 前景：被拯救的点 (按真实类别上色，大点，不透明)
        # 我们只画被拯救点，看看它们是否从“混乱”跑到了“聚类中心”
        for i, lbl in enumerate(unique_labels):
            # 该类别下的 saved 样本
            idx = saved_indices[labels[saved_indices] == lbl]
            if len(idx) > 0:
                ax.scatter(tsne_data[idx, 0], tsne_data[idx, 1], 
                           c=[palette[i]], s=50, edgecolors='black', linewidth=0.5, 
                           label=f'Saved Class {lbl}', zorder=10)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticks([]); ax.set_yticks([])

    plot_on(axes[0], tsne_sp, f"Branch B (Spectra) Features\n(Highlighting {len(saved_indices)} samples misclassified here)")
    plot_on(axes[1], tsne_fusion, f"Fusion Features\n(Same samples, now correctly classified)")
    
    # 图例 (只在右图放一个)
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="True Class of Saved Samples")
    
    plt.tight_layout()
    save_path = f"correction_analysis_{args.target_snr}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[SUCCESS] Saved analysis plot to: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, default='/home/kk/D00/data_set_5e6')
    parser.add_argument('--exp_name', type=str, default='F_T_small_gate_mulitloss_1119')
    parser.add_argument('--model', type=str, default='F_T_small_gate_mulitloss_1119')
    parser.add_argument('--class_num', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--max_samples', type=int, default=2000) # 建议跑多一点样本以找到足够的 saved cases
    parser.add_argument('--target_snr', type=str, default="pure")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=32) 
    parser.add_argument('--ratio', type=float, default=1.0)

    args = parser.parse_args()
    run_analysis(args)