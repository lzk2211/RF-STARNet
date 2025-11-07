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
        snr = 'pure' if floader == 'data_set_5e6' else floader.rsplit("5e6_", 1)[1]

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
        # 注意：为了兼容当前模型的输入通道数(in_channels=1)，这里将 L 与 S 沿宽度维度拼接，形成 [1, H, 2W]
        # 若未来希望改为 2 通道输入，可将下面的 dim=2 改为 dim=0，并相应调整模型第一层卷积的 in_channels=2
        # if _HAS_RPCA and rpca_decompose is not None:
        #     try:
        #         L, S = rpca_decompose(image, verbose=False)
        #         # 保持数值类型
        #         L = L.to(dtype=image.dtype)
        #         S = S.to(dtype=image.dtype)
        #         # 沿宽度拼接，维度：[1, H, 2W]
        #         # image = torch.cat([L, S], dim=2)
        #     except Exception as e:
        #         # RPCA 失败则回退为原图
        #         print(f"[WARN] RPCA failed for {sample_file}: {e}. Fallback to original image.")


        return image, sample_labels, snr

def data_set_split(nw, args):

    # --- 1. 设置规定的随机数 (Seed) ---
    RANDOM_SEED = args.seed # <--- 在这里设置你想要的随机种子
    torch.manual_seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    print(f"已设置随机种子: {RANDOM_SEED}")

    label_map = {}

    data_paths = ['-20', '-15', '-10', '-5', '0', '5']
    # data_paths = ['-20', '-15', '-10', '0', '5']
    data_paths = ['_' + path + 'dB/' for path in data_paths]
    data_paths.append('/')
    data_abs_paths = [args.root_path + path for path in data_paths]

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

    # --- 初始化训练、验证和测试列表 ---
    train_dataset_paths = []
    train_labels_str = []

    val_dataset_paths = []
    val_labels_str = []

    test_dataset_paths = []
    test_labels_str = []

    # 定义比例 (6:2:2)
    TRAIN_RATIO = 0.6
    VAL_RATIO = 0.2
    TEST_RATIO = 0.2 # 确保 TRAIN_RATIO + VAL_RATIO + TEST_RATIO = 1.0

    # print("开始按 SNR 和 类别 进行 6:2:2 手动分层采样...")

    # 遍历每个信噪比（SNR）路径
    for data_path in data_abs_paths:
        # print(f"正在处理路径: {data_path}")
        # 遍历每个目标类别
        for target_class in target_classes:
            label_folder_path = os.path.join(data_path, target_class)
            
            if os.path.isdir(label_folder_path):
                # 1. 获取该SNR下、该类别的所有样本文件路径
                all_files_in_folder = os.listdir(label_folder_path)
                all_file_paths = [os.path.join(label_folder_path, f) 
                                for f in all_files_in_folder if os.path.isfile(os.path.join(label_folder_path, f))]

                if not all_file_paths:
                    print(f"  - 警告: 目录 {label_folder_path} 为空, 已跳过。")
                    continue

                # --- 2. 应用你想要的 "形式" ---
                total_samples = len(all_file_paths)

                ###########################   数据采样代码   ###########################
                total_samples = total_samples * args.ratio # 按比例取样
                ###########################   数据采样代码   ###########################

                # 创建索引列表
                indices = list(range(int(total_samples)))
                
                # 使用已设置的种子打乱索引
                random.shuffle(indices) 
                
                # 3. 计算 6:2:2 分割点
                # 我们使用 math.ceil 和 math.floor 来确保所有样本都被分配
                # 并且总和不会超过 total_samples
                
                # 60% 训练
                train_size = int(total_samples * TRAIN_RATIO) 
                # 20% 验证
                val_size = int(total_samples * VAL_RATIO)
                # 剩下的是测试集 (约 20%)
                test_size = total_samples - train_size - val_size 
                
                # 4. 按照你的逻辑进行切片
                # 注意：这里的切片顺序需要调整以匹配 6:2:2
                # 我们先分出 60% (train), 再分出 20% (val), 剩下 20% (test)
                
                train_indices = indices[:train_size]
                val_indices = indices[train_size : train_size + val_size]
                test_indices = indices[train_size + val_size:] # 剩下所有的都是测试集
                
                # (检查) 确保 test_indices 数量和 test_size 匹配
                # print(f"  - {target_class}: 总共 {total_samples} -> 
                #   Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}")
                
                # 5. 根据索引将 *文件路径* 添加到各自的列表中
                
                # 训练集
                train_dataset_paths.extend([all_file_paths[i] for i in train_indices])
                train_labels_str.extend([target_class] * len(train_indices))
                
                # 验证集
                val_dataset_paths.extend([all_file_paths[i] for i in val_indices])
                val_labels_str.extend([target_class] * len(val_indices))

                # if args.test_path in all_file_paths[0]:
                    # 测试集
                test_dataset_paths.extend([all_file_paths[i] for i in test_indices])
                test_labels_str.extend([target_class] * len(test_indices))

    print("...数据划分完成。")

    # 6. （可选）将字符串标签转换为数字标签
    train_labels = [label_map[cls_name] for cls_name in train_labels_str]
    val_labels = [label_map[cls_name] for cls_name in val_labels_str]
    test_labels = [label_map[cls_name] for cls_name in test_labels_str]


    # --- 打印结果，检查划分情况 ---
    print(f"\n数据集划分概览:")
    print(f"  - 训练集样本数: {len(train_dataset_paths)}")
    print(f"  - 训练集标签数: {len(train_labels)}")
    print(f"  - 验证集样本数: {len(val_dataset_paths)}")
    print(f"  - 验证集标签数: {len(val_labels)}")
    print(f"  - 测试集样本数: {len(test_dataset_paths)}")
    print(f"  - 测试集标签数: {len(test_labels)}")
    print(f"  - 总样本数: {len(train_dataset_paths) + len(val_dataset_paths) + len(test_dataset_paths)}")

    resize = 512
    data_transform = {
        "train": transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((resize, resize)),  # 调整大小到128x128
        ]),
        "val": transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((resize, resize)),  # 调整大小到128x128
        ]),
        "test": transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((resize, resize)),  # 调整大小到128x128
        ])
    }
    train_dataset = CustomDataset(train_dataset_paths, train_labels, transform=data_transform["train"], args=args)
    validate_dataset = CustomDataset(val_dataset_paths, val_labels, transform=data_transform["val"], args=args)
    test_dataset = CustomDataset(test_dataset_paths, test_labels, transform=data_transform["test"], args=args)

    # 创建DataLoader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=nw)
    validate_loader = torch.utils.data.DataLoader(validate_dataset, batch_size=args.batch_size, shuffle=False, num_workers=nw)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=nw)
    
    # 计算数据集大小
    train_num = len(train_dataset)
    val_num = len(validate_dataset)
    test_num = len(test_dataset)
    
    print(f"Training samples: {train_num}")
    print(f"Validation samples: {val_num}")
    print(f"Testing samples: {test_num}")

    return train_loader, validate_loader, test_loader, target_classes

