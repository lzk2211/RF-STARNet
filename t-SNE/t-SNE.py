import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import argparse
from sklearn.manifold import TSNE
import seaborn as sns

# 复用你原有的模块导入
try:
    from fun import folder_settings, switch_model
    from dataset.ZJU_dataset_pre_multiSNR import data_set_split
except ImportError:
    print("[WARN] 无法导入 fun 或 dataset 模块，请检查路径设置")

# 设置绘图风格
sns.set(style="whitegrid")

class FeatureExtractor:
    def __init__(self, model, target_layers):
        self.model = model
        self.features = {name: [] for name in target_layers}
        self.hooks = []
        
        # 1. 定义要 Hook 的模块映射 (直接定位到 Linear 层)
        # 这里的 Key 是你想画图的名字，Value 是模型中的层对象
        self.layer_map = {
            'res': model.aux_head_res,   # Hook Res分支分类头的输入
            'sp': model.aux_head_sp,     # Hook Sp分支分类头的输入
            'fusion': model.classifier   # Hook 融合分类头的输入
        }
        
        # 2. 注册 Hook
        for name in target_layers:
            if name in self.layer_map:
                module = self.layer_map[name]
                # 注册 forward_hook，获取 input (即进入分类器的特征)
                handle = module.register_forward_hook(self._get_hook(name))
                self.hooks.append(handle)
                print(f"[INFO] Hook registered for: {name} (Module: {module})")
            else:
                print(f"[WARN] Layer key '{name}' not found in layer_map.")

    def _get_hook(self, name):
        def hook(module, input, output):
            # input 是一个 tuple，通常 (tensor, )
            # 这就是进入 Linear 层之前的那个特征向量！
            feat = input[0] 
            
            # 确保是 (B, D) 格式
            feat = feat.flatten(1).detach().cpu()
            self.features[name].append(feat)
        return hook

    def remove_hooks(self):
        for handle in self.hooks:
            handle.remove()

    def get_all_features(self):
        results = {}
        for name, feat_list in self.features.items():
            if feat_list:
                results[name] = torch.cat(feat_list, dim=0).numpy()
            else:
                results[name] = None
        return results

def collect_data(args, model, test_loader, max_samples=2000):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model.eval()
    
    # 我们关注这三个关键位置
    target_keys = ['res', 'sp', 'fusion']
    extractor = FeatureExtractor(model, target_keys)
    
    all_labels = []
    # all_snrs = [] # 如果需要记录 SNR
    sample_count = 0
    target_snr = args.target_snr

    print(f"[INFO] 开始提取特征 (Max samples: {max_samples})...")
    
    with torch.no_grad():
        for i, (imgs, labels, snrs) in enumerate(test_loader):
            if sample_count >= max_samples:
                break
            
            # --- SNR 筛选逻辑 ---
            if target_snr and target_snr != 'all':
                valid_indices = [idx for idx, s in enumerate(snrs) if str(target_snr) in str(s)]
                if not valid_indices: continue
                
                imgs = imgs[valid_indices]
                if isinstance(labels, torch.Tensor):
                    labels = labels[valid_indices]
                else:
                    labels = torch.tensor([labels[idx] for idx in valid_indices])
            # --------------------

            imgs = imgs.to(device)
            _ = model(imgs) # Forward pass triggers hooks
            
            if isinstance(labels, torch.Tensor):
                labels = labels.cpu().numpy()
            all_labels.extend(labels)
            
            sample_count += imgs.size(0)
            print(f"  Processed {sample_count} samples...", end='\r')

    print(f"\n[INFO] 提取完成。共 {sample_count} 个样本。")
    extractor.remove_hooks()
    return extractor.get_all_features(), np.array(all_labels)

def plot_tsne(features_dict, labels, save_path="tsne_final.png", seed=42):
    valid_keys = [k for k, v in features_dict.items() if v is not None]
    if not valid_keys: return

    # 排序：Res -> Sp -> Fusion
    order = {'res': 0, 'sp': 1, 'fusion': 2}
    valid_keys.sort(key=lambda x: order.get(x, 99))
    
    n_plots = len(valid_keys)
    plt.figure(figsize=(6 * n_plots, 6))
    
    unique_labels = np.unique(labels)
    # 使用鲜艳的颜色
    palette = sns.color_palette("bright", len(unique_labels))
    
    for idx, key in enumerate(valid_keys):
        feat = features_dict[key]
        print(f"[INFO] Computing t-SNE for '{key}' (Shape: {feat.shape})...")
        
        # --- 关键修复：兼容 max_iter 和 n_iter ---
        try:
            # 优先尝试新版本参数 (v1.5+)
            tsne = TSNE(n_components=2, perplexity=40, max_iter=1500, 
                        init='pca', learning_rate='auto', random_state=seed)
        except TypeError as e:
            if 'max_iter' in str(e) or 'unexpected' in str(e):
                # 如果 max_iter 报错，说明是旧版本，尝试 n_iter
                tsne = TSNE(n_components=2, perplexity=40, n_iter=1500, 
                            init='pca', learning_rate='auto', random_state=seed)
            else:
                # 再次尝试，去掉 learning_rate='auto' 以防非常旧的版本
                tsne = TSNE(n_components=2, perplexity=40, n_iter=1500, 
                            random_state=seed)
        # ----------------------------------------
            
        tsne_res = tsne.fit_transform(feat)
        
        ax = plt.subplot(1, n_plots, idx + 1)
        
        for i, label_i in enumerate(unique_labels):
            indices = labels == label_i
            ax.scatter(tsne_res[indices, 0], tsne_res[indices, 1], 
                       c=[palette[i]], label=f"C{label_i}", s=20, alpha=0.7, edgecolors='none')
        
        # 标题显示对应的准确率（你需要手动填入或动态计算）
        acc_map = {'res': 'Res Branch', 'sp': 'Spectra Branch', 'fusion': 'Fusion'}
        ax.set_title(acc_map.get(key, key), fontsize=14, fontweight='bold')
        ax.set_xticks([]); ax.set_yticks([])
        
        if idx == n_plots - 1:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[SUCCESS] Saved to {os.path.abspath(save_path)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, default='/home/kk/D00/data_set_5e6', help='dataset path')
    parser.add_argument('--exp_name', type=str, default='F_T_small_gate_mulitloss_1119', help='exp name')
    parser.add_argument('--model', type=str, default='F_T_small_gate_mulitloss_1119')
    parser.add_argument('--class_num', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=32) 
    parser.add_argument('--ratio', type=float, default=1.0)
    parser.add_argument('--device', type=str, default='cuda:0')
    
    parser.add_argument('--max_samples', type=int, default=1000, help='用于t-SNE的最大样本数')
    parser.add_argument('--target_snr', type=str, default="pure", help='筛选特定SNR')
    
    args = parser.parse_args()

    # 1. Load Data
    print("Loading Datasets...")
    _, _, test_loader, target_classes = data_set_split(0, args)

    # 2. Load Model
    print(f"Loading Model: {args.model}...")
    model = switch_model(args.model, args.device, args.class_num)
    
    model_path = os.path.join('./1111', args.exp_name, 'model.pth')
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=args.device))
        print(f"[INFO] Weights loaded: {model_path}")
    else:
        print(f"[WARN] Weights not found at {model_path}, using random init.")
    
    model.to(args.device)

    # 3. Collect Data
    # 提取 Linear 层前的特征 (Res: 64, Sp: 512, Fusion: 576)
    feats, labels = collect_data(args, model, test_loader, max_samples=args.max_samples)

    # 4. Plot
    snr_tag = args.target_snr if args.target_snr else 'all'
    save_name = f"tsne_correct_hook_{snr_tag}.png"
    
    # 传入 seed 保证可复现
    plot_tsne(feats, labels, save_path=save_name, seed=args.seed)