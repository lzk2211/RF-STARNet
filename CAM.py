import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
import sys
from fun import folder_settings, switch_model


# 导入你的模型和数据处理模块
# 假设这些文件都在同一目录下，或者在 python path 中
from model1117.Fcnn_Tatt_gate_mulitloss import F_T
from dataset.ZJU_dataset_pre_multiSNR import data_set_split

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

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
        
        # gradients = gradients.unsqueeze(0)######
        # activations = activations.unsqueeze(0)######

        b, k, *dims = gradients.size()

        # Global Average Pooling on Gradients (计算权重 alpha)
        alpha = gradients.view(b, k, -1).mean(2) # (B, k)
        
        # 加权组合特征图
        weights = alpha.view(b, k, *([1]*len(dims)))
        cam = (weights * activations).sum(1, keepdim=True) # (B, 1, ...)

        # cam = cam.permute(0, 1, 3, 2)########
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
    heatmap_overlay = cv2.applyColorMap(heatmap_overlay, cv2.COLORMAP_JET)
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
    dataset = test_loader.dataset
    try:
        target_label_idx = target_classes.index(target_class_name)
    except ValueError:
        return []
    matched = []
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
    return matched

def load_sample_by_index(dataset, index):
    img, label, snr = dataset[index]
    if isinstance(label, torch.Tensor):
        label = label.item()
    return img.unsqueeze(0), label, snr

def run_cam_visualization(args, 
                          target_snr='0dB', 
                          target_class_name=None, 
                          alpha=0.5, 
                          bg_color=cv2.COLORMAP_OCEAN,
                          save_path="cam_result.png",
                          model=None):
    """
    入口函数：
    args: 包含 root_path, batch_size 等参数的对象
    target_snr: 想要可视化的信噪比，例如 "-5dB", "0dB"
    target_class_name: 想要可视化的类别名，例如 "T0001"。如果为None，默认取第一个类别。
    alpha: 透明度 (0.0 - 1.0)
    bg_color: 底图颜色风格 (cv2.COLORMAP_OCEAN, cv2.COLORMAP_VIRIDIS, etc.)
    """
    
    # 1. 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # 2. 加载数据集 (只获取 test_loader 和 类别列表)
    # nw 设置为 0 方便调试
    print("Loading Datasets...")
    _, _, test_loader, target_classes = data_set_split(0, args)
    
    if target_class_name is None:
        target_class_name = target_classes[0]
        
    # 3. 获取特定样本
    samples = list_samples_for_class_snr(test_loader, target_classes, target_snr, target_class_name)
    if not samples:
        print(f"[WARN] 没有找到类别 {target_class_name} 且 SNR={target_snr} 的样本，退出。")
        return
    print(f"[INFO] 共找到 {len(samples)} 个匹配样本，可供选择：")
    for idx, (sample_index, file_path) in enumerate(samples):
        print(f"  [{idx}] dataset_index={sample_index} file='{os.path.basename(file_path)}'")
    print("请输入你要选择的图片：可以输入上面显示的序号或文件名 (回车默认 0)：", end="", flush=True)
    user_in = sys.stdin.readline().strip()
    chosen = None
    if user_in == "":
        chosen = samples[0]
    else:
        # 优先当成序号
        if user_in.isdigit():
            ui = int(user_in)
            if 0 <= ui < len(samples):
                chosen = samples[ui]
        if chosen is None:
            # 作为文件名匹配
            for _, (sample_index, file_path) in enumerate(samples):
                if os.path.basename(file_path) == user_in:
                    chosen = (sample_index, file_path)
                    break
    if chosen is None:
        print("[WARN] 输入无效，使用第一个样本。")
        chosen = samples[0]
    sample_index, file_path = chosen
    print(f"[INFO] 已选择样本 dataset_index={sample_index} file='{os.path.basename(file_path)}'")
    dataset = test_loader.dataset
    img_tensor, label_idx, _ = load_sample_by_index(dataset, sample_index)
    img_tensor = img_tensor.to(device)
    
    # 4. 加载模型
    print(f"Loading Model: {args.model}...")
    # 实例化模型
    
    
    # 加载权重
    model_path = os.path.join('./1119', args.exp_name, 'model.pth')
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
    try:
        # 尝试定位到 res_backbone 的最后一层
        # 假设 F_cnn 结构中有 layer4 或 block
        if hasattr(model.res_backbone, 'layer4'):
             target_layer = model.res_backbone.layer4[-1]
            #  target_layer = model.spectra_att.att.layer_norm
        else:
             # 如果没有 layer4，取 children 的最后一个
             target_layer = list(model.res_backbone.children())[-1]
        
        # target_layer = model.att.layer_norm
        # target_layer = model.semantic_head
        print(f"[INFO] Target Layer set to: {target_layer}")
    except Exception as e:
        print(f"[Error] 无法自动定位 Target Layer: {e}")
        return

    # 6. 运行 CAM
    cam_engine = GradCAM(model, target_layer)
    mask = cam_engine(img_tensor, class_idx=label_idx) # 强制查看 Ground Truth 类别的热力图
    
    # 7. 图像合成与保存
    overlay, bg_img, heatmap = apply_colormap_on_image(
        img_tensor, mask, alpha=alpha, bg_colormap=bg_color
    )
    
    # 绘图
    plt.figure(figsize=(16, 8))

    # --- 顶部：一维投影（频率 & 时间） ---
    # 准备用于绘图的张量：去掉 batch 维度并在必要时对通道求均值
    x_plot = img_tensor.clone().detach().cpu().squeeze()
    # 处理可能的形状： (C, H, W) ->  (H, W)， 或 (H, W)
    if x_plot.ndim == 3:
        # 对 channel 求平均
        x_plot = x_plot.mean(dim=0)

    # 1D 投影
    freq_1d = x_plot.mean(dim=-1).numpy()
    time_1d = x_plot.mean(dim=-2).numpy()

    gs = GridSpec(2, 3, height_ratios=[1, 4])

    ax_freq = plt.subplot(gs[0, :2])
    ax_freq.plot(freq_1d)
    ax_freq.set_title('Frequency projection (mean over time)')
    ax_freq.set_xlabel('frequency bin')
    ax_freq.set_ylabel('mean amplitude')

    ax_time = plt.subplot(gs[0, 2])
    ax_time.plot(time_1d)
    ax_time.set_title('Time projection (mean over frequency)')
    ax_time.set_xlabel('time bin')
    ax_time.set_ylabel('mean amplitude')

    # --- 底部：原来的三张图 ---
    ax_bg = plt.subplot(gs[1, 0])
    ax_bg.imshow(bg_img)
    ax_bg.set_title(f"Input: {target_class_name} @ {target_snr}\n(False Color)")
    ax_bg.axis('off')

    ax_heat = plt.subplot(gs[1, 1])
    ax_heat.imshow(heatmap)
    ax_heat.set_title("Grad-CAM Heatmap")
    ax_heat.axis('off')

    ax_overlay = plt.subplot(gs[1, 2])
    ax_overlay.imshow(overlay)
    ax_overlay.set_title(f"Overlay (alpha={alpha})")
    ax_overlay.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"[SUCCESS] 结果已保存至: {os.path.abspath(save_path)}")
    # plt.show() # 如果在服务器无界面环境，请注释此行
    
    # 清理 hook
    cam_engine.remove_hooks()

if __name__ == "__main__":
    # 模拟 main.py 中的 args
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, default='/home/kk/D00/data_set_5e6', help='dataset path')
    parser.add_argument('--exp_name', type=str, default='F_T_small_gate_mulitloss_1119', help='exp name for finding model path')
    parser.add_argument('--model', type=str, default='F_T_small_gate_mulitloss_1119')
    parser.add_argument('--class_num', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=16) # 仅用于 loader 初始化，不影响 cam
    parser.add_argument('--ratio', type=float, default=1.0)
    parser.add_argument('--device', type=str, default='cuda:0')
    
    args = parser.parse_args()

    model = switch_model(args.model, args.device, args.class_num)

    # 调用函数进行可视化
    # 1. 修改透明度 alpha (0.1 - 0.9)
    # 2. 修改底图颜色 bg_color (cv2.COLORMAP_OCEAN, cv2.COLORMAP_VIRIDIS, cv2.COLORMAP_INFERNO)
    run_cam_visualization(
        args, 
        target_snr='pure',       # 指定想要查看的 SNR
        target_class_name='T0010', # 指定想要查看的类别
        alpha=0.4,               # 透明度：0.4 代表 heatmap 占 40%，底图占 60%
        bg_color=cv2.COLORMAP_OCEAN, # 蓝色调底图
        save_path="result_cam.png",
        model = model
    )

