import sys
import os
import random

import torch
import torch.nn as nn

from tqdm import tqdm

import argparse
import numpy as np
import json

# from dataset_vector import data_set_split
# from dataset import data_set_split
from dataset.ZJU_dataset_pre import data_set_split
try:
    from dataset.ZJU_DroneRFb import DroneRFb_data_set_split
except Exception:
    DroneRFb_data_set_split = None

## Delay heavy visualization imports to TSNE block to avoid ABI issues at startup

from fun import folder_settings, switch_model
from train_val_test import *
# from model.ResNet import ExtendedResNet

def set_seed(seed: int):
    """Set random seeds for reproducibility before model initialization."""
    try:
        os.environ["PYTHONHASHSEED"] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Make cuDNN and PyTorch ops deterministic where possible
        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception:
            pass
        try:
            # Full determinism (may impact performance; warn_only avoids hard errors on unsupported ops)
            torch.use_deterministic_algorithms(True, warn_only=True)
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
        except Exception:
            pass
    except Exception as e:
        print(f"[seed] Failed to set full determinism: {e}")

def main():
    # /root/miniconda3/bin/python /root/autodl-tmp/lzk/test/main.py --model_path 'ANN22_model.pth' --Train True --Test True --name ANN22
    parser = argparse.ArgumentParser(description="Train a localization model")
    parser.add_argument('--epochs', type=int, default=250, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    # parser.add_argument('--runs', type=str, default='runs_Resnet10_model', help='Loss runing graph')
    parser.add_argument('--Train', type=bool, default=True, help='Switch wheater to train')
    parser.add_argument('--resume', type=bool, default=False, help='train from checkpoint')
    parser.add_argument('--Test', type=bool, default=True, help='Test the test_loader use trained model')
    parser.add_argument('--TSNE', type=bool, default=False, help='The t-SNE Visualisation')

    parser.add_argument('--class_num', type=int, default=10, help='class number')
    parser.add_argument('--model', type=str, default='ResInc_GAtt', help='switch model')
    parser.add_argument('--image_path', type=str, default='/root/autodl-tmp/data_set_5e6_-5dB/', help='dataset path')
    # parser.add_argument('--image_path', type=str, default='/media/lab125/SSD11/D00/data_set_5e6_-5dB/', help='dataset path')
    # parser.add_argument('--image_path', type=str, default='/home/lab125/kk/S3R/Data/', help='dataset path')
    parser.add_argument('--exp_name', type=str, default='-5dB_ResIncbase_GAtt_1', help='the name of the experiment')
    parser.add_argument('--Dim', type=str, default=None, help='Dimensionality of the input data, 2 for 2D images, 1 for 1D signals')

    parser.add_argument('--config', type=str, default=None, help='配置文件路径，用于加载预设配置')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use for training/testing')
    # parser.add_argument('--num_samples', type=int, default=10000, help='Number of samples to use for training')

    args = parser.parse_args()
    print(args)

    # Set seeds BEFORE creating model/dataloaders so that parameter init is reproducible
    set_seed(int(getattr(args, 'seed', 42)))

    folder_path = './1030/' + args.exp_name
    if not os.path.exists(folder_path):# if the folder does not exist, than create it
        os.makedirs(folder_path)

    if args.config is None and args.Train:
        # 将args转换为字典(如果不是字典)
        if not isinstance(args, dict):
            config_dict = vars(args)
        else:
            config_dict = args
        # 保存为JSON文件
        config_path = os.path.join(folder_path, "config.json")
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=4)
    elif args.config and args.Test:
        with open(args.config, 'r') as f:
            loaded_config = json.load(f)
        # 更新默认参数
        for key, value in loaded_config.items():
            if key != 'config':  # 避免覆盖config参数本身
                setattr(args, key, value)

    device, nw, table_data = folder_settings(args, folder_path)



    # Update the model to fit your input size and number of classes
    model = switch_model(args.model, device, args.class_num)
    

    if 'data_set' in args.image_path:
        train_loader, validate_loader, test_loader, target_classes = data_set_split(nw, args)
    elif 'S3R' in args.image_path:
        if DroneRFb_data_set_split is None:
            raise ImportError("DroneRFb_data_set_split not available; ensure dataset/ZJU_DroneRFb.py is importable.")
        train_loader, validate_loader, test_loader, target_classes = DroneRFb_data_set_split(nw, args)
    print(target_classes)
    
    if args.Train:
        train(args, folder_path, model, device, train_loader, validate_loader)


    if args.Test:
        # acc, precision, recall, f1, macro_ap, micro_ap = 
        test(args, folder_path, model, device, test_loader, target_classes)

        # save_results(folder_path, acc, precision, recall, f1, macro_ap, micro_ap, table_data)


    if args.TSNE:
        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from matplotlib import colormaps  # 注意替代 cm
        print('Testing')
        model_path = folder_path + '/model.pth'
        assert os.path.exists(model_path), "cannot find {} file".format(model_path)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)

        model.eval()
        all_features = []
        all_labels = []
        with torch.no_grad():
            for test_data in tqdm(test_loader):  # validate_loader test_loader
                test_images, test_labels = test_data
                features = model(test_images.to(device), return_features=True)
                # 假设模型的倒数第二层是特征提取层
                features = features.cpu().numpy()
                all_features.append(features)
                all_labels.extend(test_labels.cpu().numpy())

        all_features = np.concatenate(all_features, axis=0)
        # Reshape all_features to have 2 dimensions
        all_features = all_features.reshape(all_features.shape[0], -1)
        all_labels = np.array(all_labels)
        
        # 单标签场景：将 one-hot 标签转换为类别索引
        all_labels = np.argmax(all_labels, axis=1)
        numeric_labels = all_labels
        unique_combinations = target_classes

        # 使用t-SNE进行降维
        tsne = TSNE(n_components=2, random_state=42)# 将高维数据降到二维
        X_tsne = tsne.fit_transform(all_features)

        # 颜色映射（单标签）
        cmaps = 'viridis'
        colors = colormaps[cmaps].resampled(len(target_classes))  # 替代 get_cmap

        # 绘图
        plt.figure(figsize=(12, 10))
        
        # 单标签场景：原始绘图方式
        scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=numeric_labels, cmap=colors, alpha=0.6)
        plt.colorbar(scatter)
        plt.title('t-SNE Visualization - Single-label Classification')
        plt.xlabel('t-SNE Feature 1')
        plt.ylabel('t-SNE Feature 2')
        
        # 自定义图例
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label=label,
                markerfacecolor=colors(i), markersize=10)
            for i, label in enumerate(target_classes)
        ]
        plt.legend(handles=legend_elements, ncol=1)

        # 保存图像
        plt.tight_layout()
        plt.savefig(os.path.join(folder_path, 'tsne_plot.png'), dpi=300, bbox_inches='tight')
        # plt.savefig(os.path.join(folder_path, 'tsne_plot.pdf'), bbox_inches='tight')  # 同时保存PDF版本
        print(f"t-SNE图已保存到: {folder_path}")
        # plt.show()
        
        # 额外创建一个显示各个单独标签分布的图（单标签场景）
        plt.figure(figsize=(15, 4))
        
        for class_idx, class_name in enumerate(target_classes):
            plt.subplot(1, len(target_classes), class_idx + 1)
            
            # 找到属于此类别的样本
            is_class = all_labels == class_idx
            not_class = all_labels != class_idx
            
            # 绘制不属于此类别的样本（灰色）
            plt.scatter(X_tsne[not_class, 0], X_tsne[not_class, 1], 
                      c='lightgray', alpha=0.3, s=20, label='Without')
            
            # 绘制属于此类别的样本（彩色）
            plt.scatter(X_tsne[is_class, 0], X_tsne[is_class, 1], 
                      c='red', alpha=0.8, s=30, label='With')
            
            plt.title(f'{class_name}\n({is_class.sum()} samples)')
            plt.xlabel('t-SNE 1')
            plt.ylabel('t-SNE 2')
            if class_idx == 0:
                plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(folder_path, 'tsne_individual_classes.png'), dpi=300, bbox_inches='tight')
        print(f"各类别分布图已保存到: {os.path.join(folder_path, 'tsne_individual_classes.png')}")
        # writer = SummaryWriter(folder_path + '/graph')
        # writer.add_graph(model,val_images.to(device))
        # writer.close()



if __name__ == '__main__':
    main()

