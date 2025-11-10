import torch.nn as nn
import torch
from torchsummary import summary
from thop import profile
import math

class ECA1d(nn.Module):
    """
    1D 版本的 Efficient Channel Attention (ECA-Net) 模块
    
    论文: "ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks"
    链接: https://arxiv.org/abs/1910.03151
    
    它使用一个自适应核大小的 1D 卷积来代替 SENet 中的全连接层，
    实现了更高效、更轻量的通道注意力。
    """
    def __init__(self, in_channel, gamma=2, b=1):
        """
        初始化 ECA1d 模块
        
        :param in_channel: int, 输入特征图的通道数 (C)
        :param gamma: int/float, 用于计算 k_size 的超参数
        :param b: int/float, 用于计算 k_size 的超参数
        """
        super(ECA1d, self).__init__()
        
        # --- 1. Squeeze 操作 ---
        # 全局平均池化，将 (B, C, L) -> (B, C, 1)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # --- 2. 自适应 k_size 计算 ---
        # 根据通道数 C 动态计算 1D 卷积的核大小 k_size
        # 论文中的公式: k = |(log2(C) + b) / gamma|_odd
        t = int(abs((math.log2(in_channel) + b) / gamma))
        k_size = t if t % 2 else t + 1  # 确保 k_size 是奇数
        
        # --- 3. Excitation 操作 ---
        # 使用 1D 卷积实现跨通道的局部信息交互
        # padding = (k_size - 1) // 2 是为了保持卷积前后的 "长度" (在这里是通道数C) 不变
        self.conv = nn.Conv1d(
            in_channels=1, 
            out_channels=1, 
            kernel_size=k_size, 
            padding=(k_size - 1) // 2, 
            bias=False
        )
        
        # --- 4. 激活函数 ---
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        前向传播
        :param x: 输入张量, shape (B, C, L)
        :return: 经过注意力加权后的张量, shape (B, C, L)
        """
        # x 原始 shape: (B, C, L)
        b, c, l = x.size()
        
        # --- Squeeze ---
        # y shape: (B, C, 1)
        y = self.avg_pool(x)
        
        # --- Excitation ---
        # 为了让 1D 卷积在 "通道维度" C 上进行，而不是在 "长度维度" L 上进行
        # 我们需要进行两次转置 (Permute):
        
        # (B, C, 1) -> (B, 1, C)  
        # 把通道 C 当作 1D 卷积的 "长度"
        y = y.permute(0, 2, 1)
        
        # 1D 卷积在 C 维度上操作
        # (B, 1, C) -> (B, 1, C)
        y = self.conv(y)
        
        # 转置回来
        # (B, 1, C) -> (B, C, 1)
        y = y.permute(0, 2, 1)
        
        # --- Sigmoid 激活 ---
        # 得到每个通道的注意力权重 (0 到 1 之间)
        # y shape: (B, C, 1)
        y = self.sigmoid(y)
        
        # --- Rescale (Fscale) ---
        # (B, C, L) * (B, C, 1) -> (B, C, L)
        # 利用 PyTorch 的广播机制 (Broadcasting)，将每个通道的权重
        # 乘到该通道的所有时间点上
        return x * y
    
# ---------------------------------
# 模块 0: ShiftwiseConv1d (无变化)
# ---------------------------------
class ShiftwiseConv1d(nn.Module):
    """
    1D 版本的 ShiftwiseConv (SWConv)
    基于 https://github.com/lidc54/shift-wiseConv (swconv_v1.py) 的逻辑
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, 
                 dilation=1, groups=1, bias=True, n_div=4):
        super(ShiftwiseConv1d, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.n_div = n_div
        self.k = kernel_size // 2
        
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, 
            stride=stride, padding=padding, dilation=dilation, 
            groups=groups, bias=bias
        )
        
        if self.in_channels < self.n_div:
            self.n_div = self.in_channels

    def forward(self, x):
        # x shape: (B, C, L)
        B, C, L = x.shape
        
        n_div = self.n_div
        if C < n_div:
            n_div = C
            
        if self.k == 0 or n_div <= 1:
            return self.conv(x)
            
        chunk_size = C // n_div
        chunks_list = [chunk_size] * (n_div - 1)
        chunks_list.append(C - sum(chunks_list))
        
        x_chunks = torch.split(x, chunks_list, dim=1)
        
        shifted_chunks = []
        for i, chunk in enumerate(x_chunks):
            if i == 0:
                shifted_chunks.append(chunk)
            else:
                shift_amount = i * self.k
                shifted_chunk = torch.roll(chunk, shifts=shift_amount, dims=2)
                shifted_chunks.append(shifted_chunk)
        
        x_shifted = torch.cat(shifted_chunks, dim=1)
        out = self.conv(x_shifted)
        return out

# ---------------------------------
# 模块 1.1-SW: SW_BasicBlock1D (无变化)
# ---------------------------------
class SW_BasicBlock1D(nn.Module):
    """ BasicBlock1D, 但使用 ShiftwiseConv1d 替换 3x3 卷积 """
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, groups=1, **kwargs):
        super(SW_BasicBlock1D, self).__init__()
        n_div = kwargs.get('n_div', 4)

        self.conv1 = ShiftwiseConv1d(in_channels=in_channel, out_channels=out_channel,
                                     kernel_size=3, stride=stride, padding=1, bias=False, 
                                     groups=groups, n_div=n_div)
        self.bn1 = nn.BatchNorm1d(out_channel)
        self.relu = nn.ReLU()
        
        self.conv2 = ShiftwiseConv1d(in_channels=out_channel, out_channels=out_channel,
                                     kernel_size=3, stride=1, padding=1, bias=False, 
                                     groups=groups, n_div=n_div)
        self.bn2 = nn.BatchNorm1d(out_channel)
        self.downsample = downsample

        self.eca = ECA1d(out_channel)

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out = self.eca(out)
        out += identity
        out = self.relu(out)
        return out

# ---------------------------------
# 模块 2.1-SW: SW_Bottleneck1D (无变化)
# ---------------------------------
class SW_Bottleneck1D(nn.Module):
    """ Bottleneck1D, 但使用 ShiftwiseConv1d 替换 3x3 卷积 """
    expansion = 4 

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, groups=1, width_per_group=64, **kwargs):
        super(SW_Bottleneck1D, self).__init__()
        n_div = kwargs.get('n_div', 4)
        
        width = int(out_channel * (width_per_group / 64.)) * groups
        
        self.conv1 = nn.Conv1d(in_channels=in_channel, out_channels=width,
                               kernel_size=1, stride=1, bias=False, groups=groups)
        self.bn1 = nn.BatchNorm1d(width)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = ShiftwiseConv1d(in_channels=width, out_channels=width,
                                     kernel_size=3, stride=stride, padding=1, bias=False, 
                                     groups=groups, n_div=n_div)
        self.bn2 = nn.BatchNorm1d(width)
        
        self.conv3 = nn.Conv1d(in_channels=width, out_channels=out_channel * self.expansion,
                               kernel_size=1, stride=1, bias=False, groups=groups)
        self.bn3 = nn.BatchNorm1d(out_channel * self.expansion)
        
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out += identity
        out = self.relu(out)
        return out


# ---------------------------------
# 模块 3: ResNet1D (### forward 方法已修改 ###)
# ---------------------------------
class ResNet1D(nn.Module):

    def __init__(self,
                 block,
                 blocks_num,
                 num_classes=1000,
                 include_top=True,
                 groups=1,
                 width_per_group=64,
                 base_channels=64,
                 width_mult: float = 1.0,
                 stem_channels: int = None,
                 dropout: float = 0.0,
                 n_div: int = 4): 
        super(ResNet1D, self).__init__()
        self.include_top = include_top
        self.in_channel = stem_channels if stem_channels is not None else base_channels
        self.groups = groups
        self.width_per_group = width_per_group
        self.base_channels = base_channels
        self.width_mult = float(width_mult)
        self.dropout_p = float(dropout)
        self.n_div = n_div 

        self.conv1 = nn.Conv1d(1, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        def _make_divisible(v: int, divisor: int) -> int:
            if divisor <= 1:
                return int(v)
            v = int(v)
            return max(divisor, (v + divisor - 1) // divisor * divisor)

        stage_base = [1, 2, 4, 8]
        stage_channels = [
            _make_divisible(int(self.base_channels * b * self.width_mult), self.groups)
            for b in stage_base
        ]
        
        self.layer1 = self._make_layer(block, stage_channels[0], blocks_num[0])
        self.layer2 = self._make_layer(block, stage_channels[1], blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, stage_channels[2], blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, stage_channels[3], blocks_num[3], stride=2)

        self.dropout = nn.Dropout(p=self.dropout_p) if self.dropout_p and include_top else nn.Identity()
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool1d(1)
            final_channels = stage_channels[3] * block.expansion
            self.fc = nn.Linear(final_channels, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

        self.thresholds = nn.Parameter(torch.ones(num_classes) * 0.5)

    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channel, channel * block.expansion, kernel_size=1,
                          stride=stride, bias=False, groups=self.groups),
                nn.BatchNorm1d(channel * block.expansion))

        layers = []
        layers.append(block(self.in_channel,
                            channel,
                            downsample=downsample,
                            stride=stride,
                            groups=self.groups,
                            width_per_group=self.width_per_group,
                            n_div=self.n_div)) 
        
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channel,
                                channel,
                                groups=self.groups,
                                width_per_group=self.width_per_group,
                                n_div=self.n_div)) 

        return nn.Sequential(*layers)

    # ----------------------------------------------------
    # ### 这里是您请求的修改 ###
    # ----------------------------------------------------
    def forward(self, x):
        
        # ### 新增 ###: 根据您的要求，处理 4D (B, C, H, W) 输入
        # 假设输入是 (B, 1, 512, 512)，将其转换为 (B, 1, 512)
        # 如果输入已经是 3D (B, C, L)，则跳过此步骤
        if x.dim() == 4:
            x = x.mean(dim=-1)
        
        # 现在的 x 应该是 (B, 1, L)，例如 (B, 1, 512)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.dropout(x)
            x = self.fc(x)

        return x
    # ----------------------------------------------------
    # ### 修改结束 ###
    # ----------------------------------------------------

# ---------------------------------
# 模块 4: 工厂函数 (无变化)
# ---------------------------------
def Res_Inc_Net(
    block,  
    num_classes: int = 1000,
    include_top: bool = True,
    layers: tuple[int, int, int, int] = (1, 1, 1, 1),
    groups: int = 32,
    base_channels: int = 64,
    width_mult: float = 1.0,
    stem_channels: int = None,
    dropout: float = 0.5,
    n_div: int = 4 
):
    return ResNet1D(
        block,  
        list(layers),
        num_classes=num_classes,
        include_top=include_top,
        groups=groups,
        width_per_group=4, 
        base_channels=base_channels,
        width_mult=width_mult,
        stem_channels=stem_channels,
        dropout=dropout,
        n_div=n_div 
    )

def SW_Res_Inc_Net_base(num_classes: int = 1000, include_top: bool = True, n_div: int = 4):
    """ResNet-18 结构 (SW_BasicBlock1D, [2,2,2,2])"""
    return Res_Inc_Net(SW_BasicBlock1D, num_classes=num_classes, include_top=include_top,
        layers=(2, 2, 2, 2), base_channels=64, dropout=0.5, n_div=n_div)

def SW_Res_Inc_Net_large(num_classes: int = 1000, include_top: bool = True, n_div: int = 4):
    """ResNet-34 结构 (SW_BasicBlock1D, [3,4,6,3])"""
    return Res_Inc_Net(SW_BasicBlock1D, num_classes=num_classes, include_top=include_top,
        layers=(3, 4, 6, 3), base_channels=64, dropout=0.5, n_div=n_div)

def SW_Res_Inc_Net_50(num_classes: int = 1000, include_top: bool = True, n_div: int = 4):
    """ResNet-50 结构 (SW_Bottleneck1D, [3,4,6,3])"""
    return Res_Inc_Net(SW_Bottleneck1D, num_classes=num_classes, include_top=include_top,
        layers=(3, 4, 6, 3), base_channels=64, dropout=0.5, n_div=n_div)

def SW_Res_Inc_Net_101(num_classes: int = 1000, include_top: bool = True, n_div: int = 4):
    """ResNet-101 结构 (SW_Bottleneck1D, [3,4,23,3])"""
    return Res_Inc_Net(SW_Bottleneck1D, num_classes=num_classes, include_top=include_top,
        layers=(3, 4, 23, 3), base_channels=64, dropout=0.5, n_div=n_div)

if False:
    from torchsummary import summary
    from thop import profile
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # ### 修改 ###: 使用您指定的 4D 输入
    # (Batch, Channels, Height, Width) 或 (B, C, F, T)
    inputs = torch.randn(1, 1, 512, 512).to(device) 

    print("\n--- SW-Conv ResNet-50 (SW_Bottleneck1D, n_div=4) ---")
    model_sw_50 = SW_Res_Inc_Net_base(num_classes=10, include_top=True, n_div=4).to(device)
    # 使用 4D 输入测试
    flops_sw_50, params_sw_50 = profile(model_sw_50, (inputs,), verbose=False)
    print(f'Flops: {flops_sw_50/1e9:.2f} G, Params: {params_sw_50/1e6:.2f} M')
    print(f'  (输入 {tuple(inputs.shape)} -> .mean(dim=-1) -> {tuple(inputs.mean(dim=-1).shape)})')

    