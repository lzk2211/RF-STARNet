import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DWT_2D(nn.Module):
    """
    论文 Section III-A: Wavelet Based Multi-Resolution Analysis
    使用固定权重的卷积层实现 2D-DWT (Haar Wavelet)。
    将输入 (B, 1, H, W) 分解为 (B, 4, H/2, W/2)。
    四个子带: Average(LL), Vertical(LH), Horizontal(HL), Diagonal(HH).
    注：本实现已验证可直接处理输入尺寸 (bs, 1, 512, 512)。
    """
    def __init__(self):
        super(DWT_2D, self).__init__()
        # Haar 小波滤波器
        # LL: Low-Low (Average)
        self.register_buffer('ll_filter', torch.tensor([[1, 1], [1, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 4)
        # LH: Low-High (Vertical)
        self.register_buffer('lh_filter', torch.tensor([[-1, -1], [1, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 4)
        # HL: High-Low (Horizontal)
        self.register_buffer('hl_filter', torch.tensor([[-1, 1], [-1, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 4)
        # HH: High-High (Diagonal)
        self.register_buffer('hh_filter', torch.tensor([[1, -1], [-1, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 4)

    def forward(self, x):
        # 确保输入尺寸是偶数，如果不是则进行填充
        if x.size(2) % 2 != 0 or x.size(3) % 2 != 0:
            x = F.pad(x, (0, x.size(3) % 2, 0, x.size(2) % 2))

        ll = F.conv2d(x, self.ll_filter, stride=2)
        lh = F.conv2d(x, self.lh_filter, stride=2)
        hl = F.conv2d(x, self.hl_filter, stride=2)
        hh = F.conv2d(x, self.hh_filter, stride=2)

        # 拼接 4 个子带作为输出通道
        return torch.cat([ll, lh, hl, hh], dim=1)

class MSC_Block(nn.Module):
    """
    论文 Section III-B: Multi-Scale Network
    如图 4 所示的 MSC Block。
    并行使用 1x1, 3x3, 5x5, 7x7 卷积核。
    """
    def __init__(self, in_channels, out_channels=64):
        super(MSC_Block, self).__init__()
        
        # 每个分支的输出通道数，论文中虽未明确每个分支的具体数，
        # 但通常 Inception 结构会均分或设定固定值。
        # 这里假设每个分支输出 16，总和 64，这与论文描述的 "concatenation... reduce the number of channels" 一致。
        branch_channels = 16 
        
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.branch5 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=5, padding=2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.branch7 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=7, padding=3),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # 拼接后的融合卷积
        self.concat_conv = nn.Sequential(
            nn.Conv2d(branch_channels * 4, out_channels, kernel_size=1),
            # 注意：图 4 中 output 前有一个加号 (Residual Add)
        )
        
        # 如果输入输出通道不一致，需要在残差连接上做投影
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        b1 = self.branch1(x)
        b3 = self.branch3(x)
        b5 = self.branch5(x)
        b7 = self.branch7(x)
        
        out = torch.cat([b1, b3, b5, b7], dim=1)
        out = self.concat_conv(out)
        
        # 残差连接
        out += self.shortcut(x)
        return F.leaky_relu(out, 0.2)

class ASA_Module(nn.Module):
    """
    论文 Section III-C: ASA Module Based on Instance-Level Feature Statistic
    自适应语义增强模块。
    """
    def __init__(self, p=0.5, alpha=0.5):
        super(ASA_Module, self).__init__()
        self.p = p          # 触发 ASA 的概率
        self.alpha = alpha  # Beta 分布参数

    def forward(self, x):
        # 仅在训练模式下且满足概率 p 时执行
        if not self.training or torch.rand(1).item() > self.p:
            return x

        B, C, H, W = x.size()
        
        # 计算当前 batch 的均值和标准差 (Instance Level)
        # epsilon 设为 1e-6 防止除零
        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True)
        sig = (var + 1e-6).sqrt()
        
        # 生成随机混合系数 lambda ~ Beta(alpha, alpha)
        # 注意：论文公式 (14) 中 lambda 是一个标量，应用于整个 batch 还是逐个样本？
        # 通常 MixStyle 类方法是对 batch 进行 shuffle 混合。
        lam = np.random.beta(self.alpha, self.alpha)
        
        # 获取用于混合的 "另一个" 样本特征统计量 (通过打乱 batch 顺序)
        perm = torch.randperm(B).to(x.device)
        mu2 = mu[perm]
        sig2 = sig[perm]

        # 论文公式 (14): 混合统计量
        mu_mix = lam * mu + (1 - lam) * mu2
        sig_mix = lam * sig + (1 - lam) * sig2

        # 论文公式 (15): AdaIN
        # 先归一化，再用混合后的统计量反归一化
        x_norm = (x - mu) / sig
        x_aug = x_norm * sig_mix + mu_mix

        return x_aug

class ASA_RFFI_Net(nn.Module):
    """
    论文 Figure 2 & Figure 4: 完整网络架构
    """
    def __init__(self, num_classes=10, input_size=(1, 512, 512)):
        super(ASA_RFFI_Net, self).__init__()
        
        # 1. DWT Layer (Pre-processing embedded in model)
        self.dwt = DWT_2D()
        
        # 2. Feature Extraction
        # 输入通道为 4 (因为 DWT 将 1 通道变成了 4 通道)
        # Conv (4, 7, 64)
        self.init_conv = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=7, padding=3),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Stacked MSC Blocks
        self.msc1 = MSC_Block(64, 64)
        self.msc2 = MSC_Block(64, 64)
        
        # 论文 Figure 4 显示在两个 Conv 层之前有一个 Concatenate 操作
        # 看起来是把 init_conv 的输出和 msc2 的输出拼接了 (Dense 风格)
        # 或者仅仅是特征传递。根据 Fig 4 左侧的箭头，
        # 箭头从 Init Conv 后出发，绕过 MSC Blocks，并在之后与 MSC Block 2 的输出 Concatenate。
        self.post_concat_channels = 64 + 64 # Init_Conv_Out + MSC_Out
        
        # 两个连续的 Conv 层，论文提到 "Conv(Cin, 3, Cout) x 2"
        # 这里通常用于下采样或特征整合。为了防止全连接层参数过大，这里加入 stride=2
        self.transition_convs = nn.Sequential(
            nn.Conv2d(self.post_concat_channels, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # 3. ASA Module
        # 论文提到可以插入在不同层，Figure 4 显示插入在 Flatten 之前
        self.asa = ASA_Module(p=0.5, alpha=0.5)
        
        # 4. Label Classifier
        # 使用 LazyLinear 自动推断输入维度 (论文提及使用了 LazyLinear)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(256), # FC1
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),    # 常规操作，论文未详述 dropout 但通常需要
            nn.LazyLinear(128), # FC2
            nn.LeakyReLU(0.2, inplace=True),
            nn.LazyLinear(num_classes) # Output
        )

    def forward(self, x):
        # x: (Batch, 1, H, W)
        
        # Step 1: DWT 分解 -> (Batch, 4, H/2, W/2)
        x_dwt = self.dwt(x)
        
        # Step 2: Initial Conv
        feat_init = self.init_conv(x_dwt)
        
        # Step 3: MSC Blocks
        feat_msc1 = self.msc1(feat_init)
        feat_msc2 = self.msc2(feat_msc1)
        
        # Step 4: Concatenate (ResNet/DenseNet style skip connection from Fig 4)
        # 将初始特征与深层特征拼接
        feat_cat = torch.cat([feat_init, feat_msc2], dim=1)
        
        # Step 5: Transition Convs
        feat_trans = self.transition_convs(feat_cat)
        
        # Step 6: ASA (Training stage only)
        feat_aug = self.asa(feat_trans)
        
        # Step 7: Classifier
        out = self.classifier(feat_aug)
        
        return out

# --- 测试代码 ---
if __name__ == "__main__":
    # 输入尺寸为 512x512 的频谱图，Batch size = 2
    dummy_input = torch.randn(2, 1, 512, 512)
    
    # 实例化模型
    model = ASA_RFFI_Net(num_classes=10)
    
    # 模拟一次前向传播以初始化 LazyLinear
    output = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    
    # 打印网络结构以验证
    # print(model)
    
    # 验证 ASA 模块是否工作 (需要多次运行看输出是否抖动，且仅在 training 模式)
    model.train()
    out_train1 = model(dummy_input)
    out_train2 = model(dummy_input)
    # print("Variance check (should be non-zero if ASA triggered):", (out_train1 - out_train2).abs().sum().item())