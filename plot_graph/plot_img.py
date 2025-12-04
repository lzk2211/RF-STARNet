import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import sys
from dataset.ZJU_dataset_pre_multiSNR import data_set_split
from fun import switch_model
from model1119.T_att import T_att_nano

def list_samples(test_loader, target_classes, target_snr, target_class_name):
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
            path_list = getattr(dataset, 'file_paths', None)
            if path_list is not None:
                matched.append((i, dataset.file_paths[i]))
            else:
                matched.append((i, f"index_{i}"))
    return matched

def load_sample(dataset, index):
    img, label, snr = dataset[index]
    if isinstance(label, torch.Tensor):
        label = label.item()
    return img.unsqueeze(0), label, snr

def save_spectrogram(img_tensor, out_path, colormap=cv2.COLORMAP_OCEAN):
    img_np = img_tensor.squeeze().cpu().numpy()
    norm = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
    uint8_img = np.uint8(norm * 255)
    colored = cv2.applyColorMap(uint8_img, colormap)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(6,5))
    plt.imshow(colored)
    plt.title('Spectrogram (False Color)')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[SAVE] 频谱图保存: {os.path.abspath(out_path)}")

def save_frequency_vector(img_tensor, out_path_img, out_path_npy):
    x = img_tensor.squeeze().cpu()
    if x.ndim == 3:  # (C,H,W) -> (H,W)
        x = x.mean(dim=0)
    freq_vec = x.mean(dim=-1).numpy()  # 平均时间得到频域向量
    np.save(out_path_npy, freq_vec)
    plt.figure(figsize=(7,3))
    plt.plot(freq_vec)
    plt.title('Frequency-Domain Vector (mean over time)')
    plt.xlabel('Frequency Bin')
    plt.ylabel('Mean Amplitude')
    plt.tight_layout()
    plt.savefig(out_path_img, dpi=300)
    plt.close()
    print(f"[SAVE] 频域向量图保存: {os.path.abspath(out_path_img)}")
    print(f"[SAVE] 频域向量数值保存: {os.path.abspath(out_path_npy)}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='选择测试集样本并保存频谱图与频域向量图')
    parser.add_argument('--root_path', type=str, default='/home/kk/D00/data_set_5e6')
    parser.add_argument('--exp_name', type=str, default='F_T_tiny_gate_mulitloss_1119')
    parser.add_argument('--model', type=str, default='F_T_tiny_gate_mulitloss_1119')
    parser.add_argument('--class_num', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--ratio', type=float, default=1.0)
    parser.add_argument('--snr', type=str, default='pure', help='目标 SNR 值，例如 pure / -5dB / 0dB')
    parser.add_argument('--class_name', type=str, default='T0010', help='目标类别名称')
    parser.add_argument('--output_dir', type=str, default='./sample_outputs', help='输出保存目录')
    parser.add_argument('--colormap', type=str, default='OCEAN', help='OpenCV 伪彩色名称: OCEAN, VIRIDIS, INFERNO, JET 等')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print('[INFO] 加载测试集...')
    _, _, test_loader, target_classes = data_set_split(0, args)

    if args.class_name not in target_classes:
        print(f'[ERROR] 指定的类别 {args.class_name} 不在数据集中: {target_classes}')
        return

    samples = list_samples(test_loader, target_classes, args.snr, args.class_name)
    if not samples:
        print(f'[WARN] 未找到类别 {args.class_name} 且 SNR={args.snr} 的样本')
        return
    print(f'[INFO] 找到 {len(samples)} 个匹配样本:')
    for idx, (ds_idx, path) in enumerate(samples):
        print(f'  [{idx}] dataset_index={ds_idx} file={os.path.basename(path)}')
    print('请输入要选择的样本序号或文件名 (回车默认0): ', end='', flush=True)
    user_in = sys.stdin.readline().strip()
    chosen = None
    if user_in == '':
        chosen = samples[0]
    else:
        if user_in.isdigit():
            ui = int(user_in)
            if 0 <= ui < len(samples):
                chosen = samples[ui]
        if chosen is None:
            for ds_idx, path in samples:
                if os.path.basename(path) == user_in:
                    chosen = (ds_idx, path)
                    break
    if chosen is None:
        print('[WARN] 输入无效，使用第一个样本')
        chosen = samples[0]

    sample_index, file_path = chosen
    print(f'[INFO] 已选择: dataset_index={sample_index} file={os.path.basename(file_path)}')

    dataset = test_loader.dataset
    img_tensor, label_idx, snr = load_sample(dataset, sample_index)
    img_tensor = img_tensor.to(device)

    # 保存频谱图
    cmap_name = args.colormap.upper()
    cv2_cmap = getattr(cv2, f'COLORMAP_{cmap_name}', cv2.COLORMAP_OCEAN)
    spectrogram_path = os.path.join(args.output_dir, 'spectrogram.png')
    save_spectrogram(img_tensor, spectrogram_path, colormap=cv2_cmap)

    # 保存频域向量图
    freq_vec_img_path = os.path.join(args.output_dir, 'frequency_vector.png')
    freq_vec_npy_path = os.path.join(args.output_dir, 'frequency_vector.npy')
    save_frequency_vector(img_tensor, freq_vec_img_path, freq_vec_npy_path)

    # 额外：送入 T_att_nano 的 stem，并将 (1,1,512,8) 上采样回 (1,1,512,512) 后保存
    try:
        model_tatt = T_att_nano(freq_bins=512, seq_len=512, num_classes=args.class_num, include_top=False).to(device)
        model_tatt.eval()
        with torch.no_grad():
            x_stem = model_tatt.stem_pool(img_tensor)  # (1,1,512,8)
            # 上采样到 (1,1,512,512)，仅沿时间维放大 64 倍
            x_up = F.interpolate(x_stem, scale_factor=(1, 64), mode='nearest')  # (1,1,512,512)
            x_up_np = x_up.squeeze().detach().cpu().numpy()
            # 归一化到 0-255 存图
            x_norm = (x_up_np - x_up_np.min()) / (x_up_np.max() - x_up_np.min() + 1e-8)
            x_uint8 = np.uint8(x_norm * 255)
            out_path_up = os.path.join(args.output_dir, 'stem_upsampled.png')
            plt.figure(figsize=(6,5))

            colored = cv2.applyColorMap(x_uint8, cv2.COLORMAP_OCEAN)
            colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)

            plt.imshow(colored)
            plt.title('Stem Output Upsampled (1,1,512,512)')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(out_path_up, dpi=300)
            plt.close()
            print(f"[SAVE] 上采样后的 stem 图保存: {os.path.abspath(out_path_up)}")
    except Exception as e:
        print(f"[WARN] 生成 stem 上采样图失败: {e}")

    print('[DONE] 所有文件已生成。')

if __name__ == '__main__':
    main()

