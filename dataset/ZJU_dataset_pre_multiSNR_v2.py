import os
import random
import torch
import math # 导入 math 库用于取整

import os
import random
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
# import torchvision.transforms.functional as F
import numpy as np
from torchvision import transforms

class CustomDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None, args=None):
        self.args = args
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform
        
        # 检查是否使用复值网络
        # self.use_complex = (hasattr(args, 'model') and 'complex' in args.model) if args else False

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # 获取当前样本的文件路径和标签（单标签）
        sample_file = self.file_paths[idx]
        sample_labels = self.labels[idx]

        parts = sample_file.split('/')
        floader = parts[-3]
        # 修正SNR提取逻辑，以应对可能的根路径（例如 'data_set_5e6'）
        if floader == os.path.basename(os.path.normpath(self.args.root_path if hasattr(self.args, 'root_path') else 'data_set_5e6')):
             snr = 'pure'
        else:
             snr = floader.rsplit("5e6_", 1)[-1] # 假设文件夹格式如 '...5e6_-20dB'
             if snr == floader: # 如果 rsplit 失败 (例如 'pure' 或其他格式)
                 # 尝试从路径中提取 'dB'
                 try:
                     snr_part = [p for p in parts if 'dB' in p or p == os.path.basename(os.path.normpath(self.args.root_path))]
                     if snr_part:
                         snr_folder_name = snr_part[0]
                         if 'dB' in snr_folder_name:
                             snr = snr_folder_name.replace('_', '').split('dB')[0] + 'dB'
                         else:
                             snr = 'pure'
                     else:
                         snr = 'unknown'
                 except Exception:
                     snr = 'unknown'
            
        # 加载单个样本图像（从文件路径读取图片）
        img_pil = Image.open(sample_file).convert('L')  # 转为灰度单通道

        # 应用 transform（若提供）
        if self.transform is not None:
            image = self.transform(img_pil)  # 期望返回形状 [1, H, W] 的张量，值域 [0,1]
        else:
            # 默认转换为张量
            image = transforms.ToTensor()(img_pil)  # [1,H,W], float32 [0,1]

        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(np.asarray(image))
   
        # 经过 RPCA 分解：得到低秩部分 L 与稀疏部分 S
        # ... (RPCA 逻辑保持注释状态)
        # if _HAS_RPCA and rpca_decompose is not None:
        #     ...


        return image, sample_labels, snr

def create_loader(paths, labels, transform, batch_size, shuffle, num_workers, args):
    """
    辅助函数：用于创建 Dataset 和 DataLoader
    """
    if not paths:
        print(f"[WARN] 尝试创建一个包含 0 个样本的 DataLoader。返回 None。")
        return None, 0
    
    # 确保标签是正确的数字格式（如果它们是字符串）
    # 在这个新逻辑中，标签应该已经是数字了
    
    dataset = CustomDataset(paths, labels, transform=transform, args=args)
    
    # 检查 DataLoader 是否真的被创建
    try:
        loader = torch.utils.data.DataLoader(dataset, 
                                             batch_size=batch_size, 
                                             shuffle=shuffle, 
                                             num_workers=num_workers)
        return loader, len(dataset)
    except Exception as e:
        print(f"[ERROR] 创建 DataLoader 失败: {e}")
        print(f"  > 样本数: {len(paths)}, 标签数: {len(labels)}")
        print(f"  > 转换: {transform}")
        return None, 0


def data_set_split(nw, args):

    # --- 1. 设置规定的随机数 (Seed) ---
    RANDOM_SEED = args.seed # <--- 在这里设置你想要的随机种子
    torch.manual_seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    print(f"已设置随机种子: {RANDOM_SEED}")

    # --- 2. 定义 SNR 路径和类别 ---
    
    # 定义信噪比（SNR）列表
    snr_list = ['-20', '-15', '-10', '-5', '0', '5']
    # 定义对应的文件夹名称
    snr_folders = ['_' + path + 'dB/' for path in snr_list]
    snr_folders.append('/') # 添加纯净信号的路径
    
    # 定义对应的（用于字典键的）SNR 名称
    snr_names = ['-20dB', '-15dB', '-10dB', '-5dB', '0dB', '5dB', 'pure']
    
    # 获取绝对路径
    data_abs_paths = [args.root_path + folder for folder in snr_folders]
    
    # 创建路径到SNR名称的映射
    path_snr_map = dict(zip(data_abs_paths, snr_names))

    # 目标类别列表
    num_classes = 10
    all_classes =   ["T0000", "T0001", "T0010", "T0011",
                    "T0100", "T0101", "T0110", "T0111",
                    "T1000", "T1001", "T1010", "T1011",
                    "T1100", "T1101", "T1110", "T1111",
                    "T10000", "T10001", "T10010", "T10011",
                    "T10100", "T10101", "T10110", "T10111", "T11000"]

    target_classes = all_classes[:num_classes]  # 根据num_classes选择前n个类别
    label_map = {cls: i for i, cls in enumerate(target_classes)}  # 映射类别到数字标签

    # --- 3. 初始化数据存储 (按SNR) ---
    # 使用字典来按 SNR 存储 训练、验证、测试 的（路径，标签）
    all_snr_train_paths = {snr: [] for snr in snr_names}
    all_snr_train_labels = {snr: [] for snr in snr_names}
    all_snr_val_paths = {snr: [] for snr in snr_names}
    all_snr_val_labels = {snr: [] for snr in snr_names}
    all_snr_test_paths = {snr: [] for snr in snr_names}
    all_snr_test_labels = {snr: [] for snr in snr_names}

    # 定义比例 (6:2:2)
    TRAIN_RATIO = 0.6
    VAL_RATIO = 0.2
    TEST_RATIO = 0.2 # 确保 TRAIN_RATIO + VAL_RATIO + TEST_RATIO = 1.0

    print("开始按 (SNR, 类别) 进行 6:2:2 分层采样...")

    # --- 4. 遍历每个SNR和每个类别，执行 6:2:2 划分 ---
    for data_path, snr_name in path_snr_map.items():
        # print(f"  正在处理 SNR: {snr_name} (路径: {data_path})")
        
        for target_class in target_classes:
            label_folder_path = os.path.join(data_path, target_class)
            
            if os.path.isdir(label_folder_path):
                # 1. 获取该SNR下、该类别的所有样本文件路径
                all_files_in_folder = os.listdir(label_folder_path)
                all_file_paths = [os.path.join(label_folder_path, f) 
                                for f in all_files_in_folder if os.path.isfile(os.path.join(label_folder_path, f))]

                if not all_file_paths:
                    # print(f"  - 警告: 目录 {label_folder_path} 为空, 已跳过。")
                    continue

                # --- 2. 应用你想要的 "形式" ---
                total_samples = len(all_file_paths)

                ###########################   数据采样代码   ###########################
                total_samples = int(total_samples * args.ratio) # 按比例取样 (使用int取整)
                ###########################   数据采样代码   ###########################

                # 创建索引列表
                indices = list(range(total_samples))
                
                # 使用已设置的种子打乱索引
                random.shuffle(indices) 
                
                # 3. 计算 6:2:2 分割点
                train_size = int(total_samples * TRAIN_RATIO) 
                val_size = int(total_samples * VAL_RATIO)
                # 剩下的是测试集 (约 20%)
                test_size = total_samples - train_size - val_size 
                
                # 4. 按照逻辑进行切片
                train_indices = indices[:train_size]
                val_indices = indices[train_size : train_size + val_size]
                test_indices = indices[train_size + val_size:] # 剩下所有的都是测试集
                
                # 5. 根据索引将 *文件路径* 和 *数字标签* 添加到各自SNR的列表中
                
                # 训练集
                all_snr_train_paths[snr_name].extend([all_file_paths[i] for i in train_indices])
                all_snr_train_labels[snr_name].extend([label_map[target_class]] * len(train_indices))
                
                # 验证集
                all_snr_val_paths[snr_name].extend([all_file_paths[i] for i in val_indices])
                all_snr_val_labels[snr_name].extend([label_map[target_class]] * len(val_indices))

                # 测试集
                all_snr_test_paths[snr_name].extend([all_file_paths[i] for i in test_indices])
                all_snr_test_labels[snr_name].extend([label_map[target_class]] * len(test_indices))

    print("...按 (SNR, 类别) 划分数据完成。")


    # --- 5. 定义数据转换 ---
    resize = 512
    data_transform = {
        "train": transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((resize, resize)),  # 调整大小到 512x512
        ]),
        "val": transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((resize, resize)),  # 调整大小到 512x512
        ]),
        "test": transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((resize, resize)),  # 调整大小到 512x512
        ])
    }
    
    # --- 6. “拼装” 3 组 Dataloaders ---
    
    train_loaders = {}
    val_loaders = {}

    # --- Loader 1: Pure (100% 纯净) ---
    print("\n--- 正在组装 Loader 1 (纯净组) ---")
    train_loader_1, train_num_1 = create_loader(
        all_snr_train_paths['pure'], all_snr_train_labels['pure'],
        data_transform["train"], args.batch_size, True, nw, args
    )
    val_loader_1, val_num_1 = create_loader(
        all_snr_val_paths['pure'], all_snr_val_labels['pure'],
        data_transform["val"], args.batch_size, False, nw, args
    )
    train_loaders['pure'] = train_loader_1
    val_loaders['pure'] = val_loader_1
    print(f"Loader 1 - 训练样本数: {train_num_1}")
    print(f"Loader 1 - 验证样本数: {val_num_1}")


    print("\n--- G正在组装 Loader 2 (加噪1组) ---")
    noisy_snrs = ['0dB', '5dB', 'pure']
    
    train_paths_2 = [p for snr in noisy_snrs for p in all_snr_train_paths[snr]]
    train_labels_2 = [l for snr in noisy_snrs for l in all_snr_train_labels[snr]]
    val_paths_2 = [p for snr in noisy_snrs for p in all_snr_val_paths[snr]]
    val_labels_2 = [l for snr in noisy_snrs for l in all_snr_val_labels[snr]]
    
    train_loader_2, train_num_2 = create_loader(
        train_paths_2, train_labels_2,
        data_transform["train"], args.batch_size, True, nw, args
    )
    val_loader_2, val_num_2 = create_loader(
        val_paths_2, val_labels_2,
        data_transform["val"], args.batch_size, False, nw, args
    )
    train_loaders['noisy_mix1'] = train_loader_2
    val_loaders['noisy_mix1'] = val_loader_2
    print(f"Loader 2 - 训练样本数: {train_num_2}")
    print(f"Loader 2 - 验证样本数: {val_num_2}")

    print("\n--- G正在组装 Loader 3 (加噪2组) ---")
    noisy_snrs = ['-10dB', '-5dB', '0dB', '5dB', 'pure']
    
    train_paths_3 = [p for snr in noisy_snrs for p in all_snr_train_paths[snr]]
    train_labels_3 = [l for snr in noisy_snrs for l in all_snr_train_labels[snr]]
    val_paths_3 = [p for snr in noisy_snrs for p in all_snr_val_paths[snr]]
    val_labels_3 = [l for snr in noisy_snrs for l in all_snr_val_labels[snr]]
    
    train_loader_3, train_num_3 = create_loader(
        train_paths_3, train_labels_3,
        data_transform["train"], args.batch_size, True, nw, args
    )
    val_loader_3, val_num_3 = create_loader(
        val_paths_3, val_labels_3,
        data_transform["val"], args.batch_size, False, nw, args
    )
    train_loaders['noisy_mix2'] = train_loader_3
    val_loaders['noisy_mix2'] = val_loader_3
    print(f"Loader 3 - 训练样本数: {train_num_3}")
    print(f"Loader 3 - 验证样本数: {val_num_3}")


    print("\n--- 正在组装 Loader 4 (全谱组) ---")
    all_snrs = snr_names # 使用所有 snr_names
    
    train_paths_4 = [p for snr in all_snrs for p in all_snr_train_paths[snr]]
    train_labels_4 = [l for snr in all_snrs for l in all_snr_train_labels[snr]]
    val_paths_4 = [p for snr in all_snrs for p in all_snr_val_paths[snr]]
    val_labels_4 = [l for snr in all_snrs for l in all_snr_val_labels[snr]]

    train_loader_4, train_num_4 = create_loader(
        train_paths_4, train_labels_4,
        data_transform["train"], args.batch_size, True, nw, args
    )
    val_loader_4, val_num_4 = create_loader(
        val_paths_4, val_labels_4,
        data_transform["val"], args.batch_size, False, nw, args
    )
    train_loaders['full_spectrum'] = train_loader_4
    val_loaders['full_spectrum'] = val_loader_4
    print(f"Loader 4 - 训练样本数: {train_num_4} (即总训练样本)")
    print(f"Loader 4 - 验证样本数: {val_num_4} (即总验证样本)")


    # --- Test Loader (统一测试集: All SNRs) ---
    print("\n--- 正在组装 Test Loader (统一测试集) ---")
    test_paths = [p for snr in all_snrs for p in all_snr_test_paths[snr]]
    test_labels = [l for snr in all_snrs for l in all_snr_test_labels[snr]]

    test_loader, test_num = create_loader(
        test_paths, test_labels,
        data_transform["test"], args.batch_size, False, nw, args
    )
    print(f"Test Loader - 测试样本数: {test_num} (即总测试样本)")

    # --- 7. 打印最终概览并返回 ---
    print(f"\n数据集划分总概览:")
    print(f"  - 总训练样本 (用于L4): {train_num_4}")
    print(f"  - 总验证样本 (用于L4): {val_num_4}")
    print(f"  - 总测试样本 (统一):   {test_num}")
    print(f"  - 总样本数 (6:2:2 划分后): {train_num_4 + val_num_4 + test_num}")

    # 返回 3 个训练 loaders (字典), 3 个验证 loaders (字典), 1 个测试 loader
    return train_loaders, val_loaders, test_loader, target_classes

