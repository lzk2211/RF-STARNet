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
from scipy.io import loadmat
from tqdm import tqdm
import torch.nn.functional as F
## Removed sklearn dependency to avoid NumPy ABI issues
try:
    # RPCA 分解（低秩+稀疏）
    from RPCA import rpca_decompose
    _HAS_RPCA = True
except Exception:
    rpca_decompose = None
    _HAS_RPCA = False

def spectrogram_process(stft_matrix):
    """
    对未归一化的STFT矩阵添加指定SNR的AWGN
    :param stft_matrix: 复数STFT矩阵 (n_freq, n_time)
    :param snr_db: 目标信噪比（分贝，默认-30dB）
    :return: 加噪后的STFT矩阵
    """    
    # 计算幅度并进行20log10处理
    magnitude = np.abs(stft_matrix)
    log_magnitude = 20 * np.log10(magnitude + 1e-8)  # 添加小常数避免log(0)
    
    # 归一化到[0,1]范围
    max_val = np.max(log_magnitude)
    min_val = np.min(log_magnitude)
    normalized = (log_magnitude - min_val) / (max_val - min_val)
    normalized = torch.from_numpy(normalized)
    tensor_float = normalized.to(torch.float32)  # 显式指定目标 dtype

    return tensor_float

def spectrogram_process_kspace(stft_matrix):
    """
    对未归一化的STFT矩阵进行处理，最终输出K空间图（2D FFT）
    :param stft_matrix: 复数STFT矩阵 (n_freq, n_time)
    :return: K空间张量（实部 + 归一化）
    """    
    # 步骤1：获取 STFT 幅度并转为 dB
    magnitude = np.abs(stft_matrix)
    log_magnitude = 20 * np.log10(magnitude + 1e-8)  # 避免log(0)

    # 步骤2：归一化到 [0, 1]
    max_val = np.max(log_magnitude)
    min_val = np.min(log_magnitude)
    normalized = (log_magnitude - min_val) / (max_val - min_val)

    # 步骤3：计算二维傅里叶变换（K空间）
    kspace = np.fft.fft2(normalized)         # 2D FFT
    kspace_shifted = np.fft.fftshift(kspace) # 将低频移到中心区域

    # 步骤4：取 K空间幅度（或相位，或实部）
    kspace_magnitude = np.abs(kspace_shifted)
    # 也可以选：np.real(kspace_shifted), np.angle(kspace_shifted)

    # 步骤5：对K空间进行对数缩放与归一化（增强可视性）
    kspace_log = np.log1p(kspace_magnitude)  # log(1 + x)
    kspace_log = (kspace_log - np.min(kspace_log)) / (np.max(kspace_log) - np.min(kspace_log))

    # 转为torch张量
    kspace_tensor = torch.from_numpy(kspace_log).to(torch.float32)

    return kspace_tensor

def save_image(img, file_path):
    """
    保存图片到指定路径
    
    参数:
        img: 可以是PIL.Image对象、numpy数组或者tensor
        file_path: 保存图片的路径，包括文件名和扩展名
    """
    # 如果是numpy数组，转换为PIL Image
    if isinstance(img, np.ndarray):
        # 确保值在0-255范围内
        if img.dtype == np.float32 or img.dtype == np.float64:
            img = (img * 255).astype(np.uint8)
        image = Image.fromarray(img)
    # 如果是tensor，先转为numpy再保存
    elif str(type(img)).find('torch.Tensor') >= 0:
        img = img.detach().cpu().numpy()
        if img.ndim == 4:  # 批量图像，取第一个
            img = img[0]
        if img.ndim == 3 and img.shape[0] in [1, 3]:  # CHW格式转为HWC
            img = np.transpose(img, (1, 2, 0))
        # 处理形状为(1,1,1)的特殊情况
        if img.shape == (1, 1, 1):
            img_value = img[0, 0, 0]
            img = np.full((100, 100), img_value, dtype=np.uint8)
        # 压缩单通道
        elif len(img.shape) == 3 and img.shape[2] == 1:
            img = img.squeeze(2)
            
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        image = Image.fromarray(img)
    else:
        image = img
    
    # 保存图片
    image.save(file_path)

class CustomDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None, args=None):
        self.args = args
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform
        
        # 检查是否使用复值网络
        self.use_complex = (hasattr(args, 'model') and 'complex' in args.model) if args else False

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # 获取当前样本的文件路径和标签（单标签）
        sample_file = self.file_paths[idx]
        sample_labels = self.labels[idx]

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

        if self.args.Dim == 'f':
            image = torch.squeeze(torch.sum(image, dim=2, keepdim=True)).unsqueeze(0) # We output a 1D vector of frequency sequence
        elif self.args.Dim == 't':
            image = torch.squeeze(torch.sum(image, dim=1, keepdim=True)).unsqueeze(0) # We output a 1D vector of time sequence

        return image, sample_labels

def data_set_split(nw, args):
    label_map = {}

    # 目标类别列表
    num_classes = args.class_num
    # all_classes = ["T0001", "T0010", "T0011", "T0100", "T0101", "T0110", "T0111", "T1000", "T1001", "T1010"]
    all_classes =   ["T0000", "T0001", "T0010", "T0011",
                    "T0100", "T0101", "T0110", "T0111",
                    "T1000", "T1001", "T1010", "T1011",
                    "T1100", "T1101", "T1110", "T1111",
                    "T10000", "T10001", "T10010", "T10011",
                    "T10100", "T10101", "T10110", "T10111", "T11000"]

    target_classes = all_classes[:num_classes]  # 根据num_classes选择前n个类别
    # target_classes = ["T0001", "T0011", "T0100", "T0101"]
    label_map = {cls: i for i, cls in enumerate(target_classes)}  # 映射类别到数字标签
    # label_map[target_classes] = [0, 1, 2, 3]  # 映射类别到数字标签
    
    # 确保必要的参数存在
    if not hasattr(args, 'image_path'):
        raise ValueError("args 必须包含 image_path 参数，指向数据集的路径")
        
    # 单标签数据集
    print("当前数据集是单标签的，使用单标签数据集")
    test_dataset_paths = []
    test_labels_paths = []
    # 提取target类别的数据集记录标签
    for target_class in target_classes:
        label_folder_path = os.path.join(args.image_path, target_class)
        if os.path.isdir(label_folder_path):
            for file_name in os.listdir(label_folder_path):
                test_dataset_paths.append(os.path.join(label_folder_path, file_name))
                test_labels_paths.append(target_class)  # 使用具体的类别名称作为标签

    # 将标签转换为数字索引并转为 one-hot
    test_labels_paths = [label_map[label] for label in test_labels_paths]
    test_labels_paths = torch.tensor(test_labels_paths)
    test_labels_paths = F.one_hot(test_labels_paths, num_classes=num_classes).float()

    resize = 512
    # 创建数据变换
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

    # 划分训练集、验证集和测试集
    # # train_ratio = 0.7
    # val_ratio = 0.5
    # test_ratio = 0.5
    
    train_ratio = 0.6
    val_ratio = 0.2
    test_ratio = 0.2

    # 确保比例之和为1
    # assert val_ratio + test_ratio == 1.0
    
    # 计算每个集合的大小
    total_samples = len(test_dataset_paths)
    train_size = int(total_samples * train_ratio)
    test_size = int(total_samples * val_ratio)
    val_size = int(total_samples * test_ratio)
    
    torch.manual_seed(42)# K-fold交叉验证时设置随机种子
    indices = list(range(total_samples))
    random.shuffle(indices)
    
    # 分割索引
    val_indices = indices[:val_size]
    test_indices = indices[val_size:test_size + val_size]
    train_indices = indices[test_size + val_size:train_size + test_size + val_size]
    
    # 创建每个数据集的文件路径和标签
    # train_files = train_dataset_paths
    # train_labels = train_labels_paths
    
    train_files = [test_dataset_paths[i] for i in train_indices]
    train_labels = [test_labels_paths[i] for i in train_indices]

    val_files = [test_dataset_paths[i] for i in val_indices]
    val_labels = [test_labels_paths[i] for i in val_indices]
    
    test_files = [test_dataset_paths[i] for i in test_indices]
    test_labels = [test_labels_paths[i] for i in test_indices]

    # 创建数据集
    train_dataset = CustomDataset(train_files, train_labels, transform=data_transform["train"], args=args)
    validate_dataset = CustomDataset(val_files, val_labels, transform=data_transform["val"], args=args)
    test_dataset = CustomDataset(test_files, test_labels, transform=data_transform["test"], args=args)

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

    # 返回所有类别标签以便后续使用
    return train_loader, validate_loader, test_loader, target_classes

def complex_spectrogram_process(stft_matrix, normalization='magnitude'):
    """
    处理复值STFT矩阵，提供多种处理选项
    :param stft_matrix: 复数STFT矩阵 (n_freq, n_time) - 可以是numpy数组或torch tensor
    :param normalization: 处理方式
        - 'none': 不处理，直接返回复值
        - 'magnitude': 返回归一化的幅度
        - 'phase': 返回相位信息
        - 'complex_norm': 复值归一化
        - 'real_imag': 分别归一化实部和虚部
    :return: 处理后的张量
    """    
    # 统一转换为numpy数组进行处理
    if isinstance(stft_matrix, torch.Tensor):
        stft_matrix = stft_matrix.numpy()
    
    if normalization == 'none':
        # 直接返回复值张量
        return torch.from_numpy(stft_matrix).to(torch.complex64)
    
    elif normalization == 'magnitude':
        # 计算幅度并进行归一化
        magnitude = np.abs(stft_matrix)
        log_magnitude = 20 * np.log10(magnitude + 1e-8)
        max_val = np.max(log_magnitude)
        min_val = np.min(log_magnitude)
        normalized = (log_magnitude - min_val) / (max_val - min_val)
        return torch.from_numpy(normalized).to(torch.float32)
    
    elif normalization == 'phase':
        # 返回相位信息
        phase = np.angle(stft_matrix)
        # 将相位从[-π, π]映射到[0, 1]
        normalized_phase = (phase + np.pi) / (2 * np.pi)
        return torch.from_numpy(normalized_phase).to(torch.float32)
    
    elif normalization == 'complex_norm':
        # 复值归一化：保持相位，归一化幅度
        magnitude = np.abs(stft_matrix)
        phase = np.angle(stft_matrix)
        
        # 归一化幅度
        max_mag = np.max(magnitude)
        min_mag = np.min(magnitude)
        normalized_mag = (magnitude - min_mag) / (max_mag - min_mag)
        
        # 重构复数
        normalized_complex = normalized_mag * np.exp(1j * phase)
        return torch.from_numpy(normalized_complex).to(torch.complex64)
    
    elif normalization == 'real_imag':
        # 分别归一化实部和虚部
        real_part = np.real(stft_matrix)
        imag_part = np.imag(stft_matrix)

        # 归一化实部/虚部（加入极小值，避免分母为0）
        eps = 1e-8
        real_max, real_min = np.max(real_part), np.min(real_part)
        denom_real = real_max - real_min
        denom_real = denom_real if denom_real != 0 else eps
        norm_real = (real_part - real_min) / denom_real
        
        imag_max, imag_min = np.max(imag_part), np.min(imag_part)
        denom_imag = imag_max - imag_min
        denom_imag = denom_imag if denom_imag != 0 else eps
        norm_imag = (imag_part - imag_min) / denom_imag
        
        # 重构复数
        normalized_complex = norm_real + 1j * norm_imag
        return torch.from_numpy(normalized_complex).to(torch.complex64)
    
    else:
        raise ValueError(f"不支持的归一化方式: {normalization}")

