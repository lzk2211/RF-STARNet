import torch
import numpy as np

# 1. SpecAugment (极其适合 RF 信号)
def spec_augment(spec, time_mask_param=30, freq_mask_param=20):
    # spec: (B, 1, F, T)
    spec = spec.clone()
    B, C, F, T = spec.shape
    
    # 频域掩码 (模拟信道干扰)
    f_mask_len = np.random.randint(0, freq_mask_param)
    f0 = np.random.randint(0, F - f_mask_len)
    spec[:, :, f0:f0+f_mask_len, :] = 0

    # 时域掩码 (模拟突发噪声/跳频丢失)
    t_mask_len = np.random.randint(0, time_mask_param)
    t0 = np.random.randint(0, T - t_mask_len)
    spec[:, :, :, t0:t0+t_mask_len] = 0
    
    return spec

# 2. Mixup (解决过拟合的神器)
def mixup_data(x, y, snr, alpha=1.0, p=0.5):
    """
    参数:
    - x: 输入张量 (B, C, L) 或 (B, ...)
    - y: 标签张量 (B,)
    - snr: 一个列表或元组，长度为 B，包含每个样本的 SNR 字符串 (例如 ["-20dB", "0dB", ...])
    - alpha: Beta 分布参数
    - p: 触发 mixup 的概率
    """
    
    # 1. 决定是否执行 mixup
    if np.random.rand() > p:
        return x, y, y, 1.0

    # 2. 定义低信噪比集合
    low_snr_set = {"-20dB", "-15dB", "-10dB", "-5dB"}
    
    batch_size = x.size(0)
    device = x.device

    # 3. 生成掩码：找出哪些样本是 Low SNR (需特殊照顾)，哪些是 High/Pure SNR (优等生)
    # 假设 snr 是字符串列表
    is_low_snr = torch.tensor([s in low_snr_set for s in snr], device=device, dtype=torch.bool)
    
    # 获取所有 "优等生" (0dB, 5dB, pure) 的索引位置
    # (~is_low_snr) 表示非低信噪比，即高信噪比
    high_snr_indices = torch.nonzero(~is_low_snr, as_tuple=False).squeeze(-1)

    # 4. 构建混合目标的索引 (target_indices)
    # 初始化一个全零索引
    target_indices = torch.zeros(batch_size, dtype=torch.long, device=device)

    # --- 情况 A: 这是一个全员低信噪比的 Batch (极端情况) ---
    # 如果当前 batch 里竟然没有一个优等生，就无法满足"低只能找高"的条件
    # 这种情况下，为了不报错，我们退化为普通的随机 mixup
    if len(high_snr_indices) == 0:
        target_indices = torch.randperm(batch_size, device=device)
    
    else:
        # --- 情况 B: 正常情况 ---
        
        # 1) 对于【高信噪比】样本 (Source is High)：
        # 规则：可以找任意样本 (Low or High)
        # 方法：直接生成全量的随机排列，填入对应位置
        full_random_indices = torch.randperm(batch_size, device=device)
        target_indices[~is_low_snr] = full_random_indices[~is_low_snr]

        # 2) 对于【低信噪比】样本 (Source is Low)：
        # 规则：只能找 0dB, 5dB, pure 的样本
        # 方法：从 high_snr_indices 中随机采样填入
        num_low = is_low_snr.sum().item()
        if num_low > 0:
            # 生成随机索引，范围是 [0, len(high_snr_indices))
            rand_select = torch.randint(high_snr_indices.size(0), (num_low,), device=device)
            # 映射回真实的 batch 索引
            selected_partners = high_snr_indices[rand_select]
            target_indices[is_low_snr] = selected_partners

    # 5. 执行 Mixup 计算
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    # 这里的混合公式： mixed = lam * 原图 + (1-lam) * 目标图
    # 也可以反过来写，不影响数学本质，只要 y 对应即可
    mixed_x = lam * x + (1 - lam) * x[target_indices]
    y_a, y_b = y, y[target_indices]

    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# 在训练循环中使用：
# inputs = spec_augment(inputs) # 先做掩码
# inputs, targets_a, targets_b, lam = mixup_data(inputs, targets)
# outputs = model(inputs)
# loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)