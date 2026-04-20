import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
import sys
import csv
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fun import folder_settings, switch_model


# 导入你的模型和数据处理模块
# 假设这些文件都在同一目录下，或者在 python path 中
# from model1117.Fcnn_Tatt_gate_mulitloss import F_T
from dataset.ZJU_dataset_pre_multiSNR import data_set_split

class GradCAM:
    def __init__(self, model, target_layer, require_1d_fix=False):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.require_1d_fix = require_1d_fix

        # 注册 Hook
        # 针对不同版本的 PyTorch，backward_hook 的行为可能不同，这里使用 full_backward_hook
        self.handles = []
        self.handles.append(target_layer.register_forward_hook(self.save_activation))
        self.handles.append(target_layer.register_full_backward_hook(self.save_gradient))

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        # grad_output[0] 是输出的梯度
        self.gradients = grad_output[0]

    def remove_hooks(self):
        for handle in self.handles:
            handle.remove()

    def __call__(self, x, class_idx=None):
        self.model.eval()
        
        # 前向传播
        logits = self.model(x) 
        
        # 兼容你的模型：如果是训练模式返回tuple，eval模式返回tensor
        # 既然这里是 eval()，通常返回 logits
        if isinstance(logits, tuple):
            logits = logits[0]

        if class_idx is None:
            class_idx = torch.argmax(logits, dim=1).item()
            print(f"[INFO] 自动选择预测概率最高的类别索引: {class_idx}")
        
        # 对应类别的得分
        score = logits[:, class_idx].squeeze()

        # 归零梯度并反向传播
        self.model.zero_grad()
        score.backward()

        # 获取梯度和特征图
        # 注意：你的 res_backbone 输出可能是 (B, C, L) 的 1D 特征
        gradients = self.gradients
        activations = self.activations
        
        # 针对 1D 特征分支的形状修正，仅在选择 spectra 层时启用
        if self.require_1d_fix:
            gradients = gradients.unsqueeze(0)
            activations = activations.unsqueeze(0)

        b, k, *dims = gradients.size()

        # Global Average Pooling on Gradients (计算权重 alpha)
        alpha = gradients.view(b, k, -1).mean(2) # (B, k)
        
        # 加权组合特征图
        weights = alpha.view(b, k, *([1]*len(dims)))
        cam = (weights * activations).sum(1, keepdim=True) # (B, 1, ...)

        # 仅在 1D 修正模式下做维度交换
        if self.require_1d_fix:
            cam = cam.permute(0, 1, 3, 2)
        # ReLU
        cam = F.relu(cam)

        # 归一化到 0-1
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-7)
        
        return cam.detach().cpu().numpy()[0, 0]

def apply_colormap_on_image(img_tensor, mask, alpha=0.5, bg_colormap=cv2.COLORMAP_OCEAN):
    """
    img_tensor: (1, 512, 512) 原始灰度图
    mask: CAM 结果，可能是 (L,) 1D 向量，也可能是 (H, W) 2D 矩阵
    alpha: 透明度，0.0-1.0，越小底图越清晰
    bg_colormap: 底图颜色风格，如 cv2.COLORMAP_OCEAN (蓝绿), cv2.COLORMAP_DEEPGREEN 等
    """
    # 1. 处理底图：转为伪彩色
    # img_tensor = img_tensor.squeeze(0)
    img_np = img_tensor.squeeze().cpu().numpy() # (512, 512)
    
    # 归一化底图到 0-255
    img_norm = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
    img_uint8 = np.uint8(255 * img_norm)
    
    # 应用伪彩色映射 (满足你不想要灰白图的要求)
    heatmap_bg = cv2.applyColorMap(img_uint8, bg_colormap)
    heatmap_bg = cv2.cvtColor(heatmap_bg, cv2.COLOR_BGR2RGB) # (512, 512, 3)
    
    # 2. 处理 Mask：调整大小并转为热力图
    # 你的模型分支如果是 1D CNN，mask 形状可能是 (128,) 或 (512,)
    # 我们需要把它拉伸成 (512, 512)
    mask = cv2.resize(mask, (img_np.shape[1], img_np.shape[0]))
    
    heatmap_overlay = np.uint8(255 * mask)
    heatmap_overlay = cv2.applyColorMap(heatmap_overlay, cv2.COLORMAP_BONE)
    heatmap_overlay = cv2.cvtColor(heatmap_overlay, cv2.COLOR_BGR2RGB)

    # 3. 叠加
    # result = alpha * heatmap + (1-alpha) * background
    overlay = np.float32(heatmap_overlay) * alpha + np.float32(heatmap_bg) * (1 - alpha)
    overlay = overlay / np.max(overlay)
    
    return overlay, heatmap_bg, heatmap_overlay

def get_specific_sample(test_loader, target_classes, target_snr, target_class_name):
    """
    遍历 DataLoader 寻找指定 SNR 和类别的第一张图片
    """
    print(f"[INFO] 正在寻找样本: SNR={target_snr}, Class={target_class_name} ...")
    
    # 这里的 dataset 是 CustomDataset
    dataset = test_loader.dataset
    
    # 查找目标类别的索引
    try:
        target_label_idx = target_classes.index(target_class_name)
    except ValueError:
        print(f"[Error] 类别 {target_class_name} 不在 target_classes 中: {target_classes}")
        return None, None

    # 遍历 dataset (直接遍历 dataset 比 loader 更容易控制索引)
    for i in range(len(dataset)):
        img, label, snr = dataset[i]
        
        # 注意 dataset[i] 返回 label 可能是 tensor 或 int
        if isinstance(label, torch.Tensor):
            label = label.item()
            
        # 检查匹配
        # 注意：dataset 中的 snr 字符串可能带 'dB' 或不带，根据你的 dataset 逻辑
        # 你的 dataset 代码: snr = 'pure' if ... else floader.rsplit("5e6_", 1)[1]
        # 通常 snr 是如 "-20dB" 或 "-20" 这样的字符串
        
        if label == target_label_idx:
            # 简单的字符串包含匹配，增强鲁棒性
            if str(target_snr) in str(snr):
                print(f"[INFO] 找到样本! Index: {i}, Path: {dataset.file_paths[i]}")
                return img.unsqueeze(0), label # 返回 (1, 1, 512, 512)
    
    print("[WARN] 未找到符合条件的样本。")
    return None, None

def list_samples_for_class_snr(test_loader, target_classes, target_snr, target_class_name):
    """
    返回符合 (类别, SNR) 条件的所有样本索引与文件名列表。
    """
    file_name = f"test_{target_class_name}_samples.csv"
    csv_folder = "./testCSV/"
    csv_file = os.path.join(csv_folder, file_name)
    matched = []

    if os.path.exists(csv_file):
        with open(csv_file, 'r', encoding='utf-8') as file:
            print("Loading")
            reader = csv.reader(file)
            for row in reader:
                idx = int(row[0])
                path = row[1]
                matched.append((idx, path))
            print(f"已读取 {csv_file}")
        return matched

    dataset = test_loader.dataset
    try:
        target_label_idx = target_classes.index(target_class_name)
    except ValueError:
        return []
    for i in range(len(dataset)):
        img, label, snr = dataset[i]
        if isinstance(label, torch.Tensor):
            label = label.item()
        if label == target_label_idx and str(target_snr) in str(snr):
            path = getattr(dataset, 'file_paths', None)
            if path is not None:
                matched.append((i, dataset.file_paths[i]))
            else:
                matched.append((i, f"index_{i}"))
    if not os.path.exists(csv_folder):
        os.makedirs(csv_folder, exist_ok=True)
    with open(csv_file, 'w', newline='', encoding='utf-8') as file:
        print("Writing")
        writer = csv.writer(file)
        writer.writerows(matched)
        print(f"已保存 {csv_file}")
    return matched

def load_sample_by_index(dataset, index):
    img, label, snr = dataset[index]
    if isinstance(label, torch.Tensor):
        label = label.item()
    return img.unsqueeze(0), label, snr

def run_cam_visualization(
                          alpha=0.5, 
                          bg_color=cv2.COLORMAP_OCEAN,
                          device=None,
                          required_tensor_label=None,
                          save_root_dir="pics",
                          save_all_pics=False
                          ):
    """
    入口函数：
    args: 包含 root_path, batch_size 等参数的对象
    target_snr: 想要可视化的信噪比，例如 "-5dB", "0dB"
    target_class_name: 想要可视化的类别名，例如 "T0001"。如果为None，默认取第一个类别。
    alpha: 透明度 (0.0 - 1.0)
    bg_color: 底图颜色风格 (cv2.COLORMAP_OCEAN, cv2.COLORMAP_VIRIDIS, etc.)
    """
    
    os.makedirs(save_root_dir, exist_ok=True)
    
    # 如果不单独生成所有样本的热力图，则在一张大图中展示所有样本的热力图，计算大图的尺寸
    if not save_all_pics:
        plotrows = len(required_tensor_label)
        plt.figure(figsize=(16, plotrows * 4))
    
    cur_plt_pos = 0 # 计数器，用于在单张大图中定位子图位置

    for pics_idx, (img_tensor, label_idx) in enumerate(required_tensor_label):
        
        # 如果生成所有样本的热力图，则每个样本单独生成一张图
        if save_all_pics:
            plt.figure(figsize=(16, 4))

        '''
            以下每张图片都单独生成cam对象的方法属于无奈之举，暂时没有找到更好的方法。
            如果生成多张图片共用一个cam，会产生污染，后一张图的热力图会受到前一张图的梯度和特征图的影响，导致结果不准确。表现为，生成A、B两图和单独生成B图，两种情况下B图的热力图不一致
            如果每生成一张图片，就调用cam的remove_hooks()方法，则所有的热力图都不准确，表现为热力图完全一样，且与输入图像无关，说明hooks被移除后，cam对象无法正确计算梯度和特征图了。
            但目前这种每次都生成cam对象的方法，会导致在生成所有图片的热力图时显存占用过高，程序被迫退出。生成少数图片，如4到5张，一般不会出现问题。
        '''
        res_cam = prepare_cam(
            switch_model(SWITCH_MODEL_ARG, args.device, args.class_num),
            TARGET_NET_PATH, 
            "res",
            device
            )
        spectra_cam = prepare_cam(
            switch_model(SWITCH_MODEL_ARG, args.device, args.class_num),
            TARGET_NET_PATH, 
            "spectra",
            device
            )
        resnet18_cam = prepare_cam(
            switch_model("resnet18", args.device, args.class_num),
            RESNET18_PATH, 
            "resnet18",
            device
            )
        cam_list = [res_cam, spectra_cam, resnet18_cam]

        for i in range(len(cam_list)):
            mask = cam_list[i](img_tensor, class_idx=label_idx) # 强制查看 Ground Truth 类别的热力图
            
            # 图像合成与保存
            overlay, bg_img, heatmap = apply_colormap_on_image(
                img_tensor, mask, alpha=alpha, bg_colormap=bg_color
            )
            if i == 0:
                cur_plt_pos += 1
                ax_bg = plt.subplot(
                    plotrows if not save_all_pics else 1, 
                    4, 
                    cur_plt_pos
                )
                ax_bg.imshow(bg_img)
                if cur_plt_pos == 1:
                    ax_bg.set_title("Input", fontsize=30, fontweight='bold', pad=20, fontfamily='Times New Roman')
                else:
                    ax_bg.set_title(None)
                ax_bg.axis('off')
            cur_plt_pos += 1
            ax_overlay = plt.subplot(
                plotrows if not save_all_pics else 1, 
                4, 
                cur_plt_pos
            )
            ax_overlay.imshow(overlay)
            if cur_plt_pos == 2:
                ax_overlay.set_title("SARNet", fontsize=30, fontweight='bold', pad=20, fontfamily='Times New Roman')
            elif cur_plt_pos == 3:
                ax_overlay.set_title("SSTA", fontsize=30, fontweight='bold', pad=20, fontfamily='Times New Roman')
            elif cur_plt_pos == 4:
                ax_overlay.set_title("2D-CNN", fontsize=30, fontweight='bold', pad=20, fontfamily='Times New Roman')
            else:
                ax_overlay.set_title(None)
            ax_overlay.axis('off')
            cam_list[i].remove_hooks()
        if save_all_pics:
            sample_save_dir = os.path.join(save_root_dir, f"{label_idx}")
            os.makedirs(sample_save_dir, exist_ok=True)
            plt.tight_layout()
            plt.savefig(os.path.join(sample_save_dir, f"{pics_idx}.png"), dpi=300)
            cur_plt_pos = 0        

    # plt.show() # 如果在服务器无界面环境，请注释此行
    if not save_all_pics:
        plt.tight_layout()
        plt.savefig(os.path.join(save_root_dir, "./result.png"), dpi=300)
        # 高质量PDF保存
        plt.savefig(os.path.join(save_root_dir, "./result.pdf"), 
                dpi=600,                    # 高分辨率
                bbox_inches='tight',        # 紧凑边框
                pad_inches=0.1,             # 内边距
                transparent=False,          # 背景透明
                format='pdf')

        # 高质量EPS保存
        plt.savefig(os.path.join(save_root_dir, "./result.eps"), 
                dpi=600,
                bbox_inches='tight',
                pad_inches=0.1,
                format='eps')
        print(f"[SUCCESS] 已保存")
        plt.close()

def prepare_samples(
        target_snr
        ):
    '''
        准备一个包含所有样本的列表
        输入：
            target_snr: 想要可视化的信噪比，例如"pure"
        输出：
            samples_list: [(target_class_name, [(sample_index, file_path), ...]), ...]
            test_loader: 用于加载样本的 DataLoader 对象
    '''
    samples_list = []
    print("Loading Datasets...")
    _, _, test_loader, target_classes = data_set_split(0, args)
    
    for target_class_name in TARGET_CLASS_NAME_LIST:
        samples = list_samples_for_class_snr(test_loader, target_classes, target_snr, target_class_name)
        if not samples:
            print(f"[WARN] 没有找到类别 {target_class_name} 且 SNR={target_snr} 的样本")
        samples_list.append((target_class_name, samples))
        print(f"[INFO] {target_class_name}共找到 {len(samples)} 个匹配样本")
    return samples_list, test_loader

def prepare_cam(
        model, 
        model_path, 
        target_kind,
        device
                ):
    '''
        准备 GradCAM 对象
        输入：
            model: 模型实例
            model_path: 权重文件路径
            target_kind: "res", "spectra", "resnet18" 之一，决定 Target Layer 的选择
            device: 设备，如 "cuda:0" 或 "cpu"
        输出：
            cam_engine: 已准备好的 GradCAM 对象
    '''
    # 加载权重
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"[INFO] 成功加载权重: {model_path}")
    else:
        print(f"[WARN] 权重文件不存在: {model_path}, 使用随机初始化参数演示。")
        
    model.to(device)
    model.eval()
    
    # 5. 定义 Target Layer
    # 针对你的 ResInc_SpAtt_Classifier
    # 我们通常想看 CNN 分支 (res_backbone) 的最后一层
    target_layer = None
    require_1d_fix = False
    try:
        if target_kind == 'spectra':
            # 光谱注意力层 (1D 特征)
            target_layer = model.spectra_att.att.layer_norm
            require_1d_fix = True
        elif target_kind == 'resnet18':
            # 二维 ResNet18 分支：优先使用封装在模型中的 resnet18 子模块
            if hasattr(model, 'resnet18') and hasattr(model.resnet18, 'layer4'):
                target_layer = model.resnet18.layer4[-1]
            elif hasattr(model, 'layer4'):
                # 如果整个模型本身就是一个 ResNet18
                target_layer = model.layer4[-1]
            else:
                raise AttributeError('未找到 ResNet18 的 layer4 模块')
            require_1d_fix = False
        else:  # 'res'
            # 1D/主干分支
            if hasattr(model, 'res_backbone'):
                if hasattr(model.res_backbone, 'layer4'):
                    target_layer = model.res_backbone.layer4[-1]
                else:
                    # 如果没有 layer4，取最后一个 child
                    target_layer = list(model.res_backbone.children())[-1]
            else:
                raise AttributeError('模型不存在 res_backbone 属性')
            require_1d_fix = False

        print(f"[INFO] Target Layer set to: {target_layer}")
    except Exception as e:
        print(f"[Error] 无法自动定位 Target Layer: {e}")
        return
    
    cam_engine = GradCAM(model, target_layer, require_1d_fix=require_1d_fix)
    return cam_engine


# 全局变量
TARGET_CLASS_NAME_LIST = [
        "T0000", "T0001", "T0010", "T0011", 
        "T0100", "T0101", "T0110", "T0111", 
        "T1000", "T1001"
        ]   # 所有类别的列表
TARGET_NET_PATH = Path("/home/kk/lzk/RF-STARNet/final_results/125/F_T_small_mulitloss_1122/model.pth")  # 要查看的模型的权重路径
RESNET18_PATH = Path("./final_results/42/compare/resnet18/model.pth")   # ResNet18的权重路径
SWITCH_MODEL_ARG = "F_T_small_mulitloss_1122"   # switch_model函数的参数
TARGET_KIND_LIST = ['res', 'spectra', 'resnet18']   # 需要生成热力图的层
SAVE_ROOT_DIR = "./pics"    # 图片保存根路径


if __name__ == "__main__":
    # 模拟 main.py 中的 args
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, default='/home/kk/D00/data_set_5e6', help='dataset path')
    parser.add_argument('--exp_name', type=str, default='F_T_small_mulitloss_1122', help='exp name for finding model path')
    parser.add_argument('--model', type=str, default='F_T_small_mulitloss_1122')
    parser.add_argument('--class_num', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=32) # 仅用于 loader 初始化，不影响 cam
    parser.add_argument('--ratio', type=float, default=1.0)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--target_snr', type=str, default='pure')
    parser.add_argument('-s', '--sample', nargs='*', type=str, required=True)
    
    args = parser.parse_args()

    if args.sample == "all":
        args.sample = ['all']

    # 1. 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 获取所有样本
    samples_list, test_loader = prepare_samples(args.target_snr)

    # 检查--sample参数有效性，并把需要画图的样本加入列表
    required_sample = []
    if args.sample[0] != 'all':
        if len(args.sample) % 2 != 0:
            print(f"[Error] --sample 参数需要为\"all\"或成对出现，例如: --sample T0001 88 T0010 20")
            sys.exit(1)
        for i in range(0, len(args.sample), 2):
            if args.sample[i] not in TARGET_CLASS_NAME_LIST:
                print(f"[Error] 无效的类别名: {args.sample[i]}. 可选类别: {TARGET_CLASS_NAME_LIST}")
                sys.exit(1)
            required_sample.append((args.sample[i], int(args.sample[i+1])))
    elif len(args.sample) > 1:
        print(f"[Error] 无效参数，{args.sample}")
        sys.exit(1)

    # 添加需要画图的样本到列表
    required_tensor_label = []
    for target_class_name, samples in samples_list:
        if args.sample[0] != 'all' and target_class_name not in [s[0] for s in required_sample]:
            continue
        for idx, sample in enumerate(samples):
            if args.sample[0] != 'all' and (target_class_name, idx) not in required_sample:
                continue
            sample_index, file_path = sample
            dataset = test_loader.dataset
            img_tensor, label_idx, _ = load_sample_by_index(dataset, sample_index)
            img_tensor = img_tensor.to(device)
            required_tensor_label.append((img_tensor, label_idx))

    # model = switch_model(args.model, args.device, args.class_num)
    
    run_cam_visualization(
        alpha=0.7,               # 透明度：0.4 代表 heatmap 占 40%，底图占 60%
        bg_color=cv2.COLORMAP_VIRIDIS, # 蓝色调底图
        device=device,
        required_tensor_label=required_tensor_label,
        save_root_dir=SAVE_ROOT_DIR,
        save_all_pics=True if args.sample[0] == 'all' else False
    )

