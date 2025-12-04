import torch.nn as nn
import torch
from torchsummary import summary
from thop import profile
import math

# ---------------------------------
# 模块 0: ShiftwiseConv1d (无变化)
# ---------------------------------
class ShiftwiseConv1d(nn.Module):
    """
    1D 版本的 ShiftwiseConv (SWConv)
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
# 模块 0.1: ECA-Net (无变化)
# ---------------------------------
class EcaChannelAttention(nn.Module):
    """ 1D 适配版 ECA-Net """
    def __init__(self, channels, gamma=2, b=1):
        super(EcaChannelAttention, self).__init__()
        t = int(abs((math.log(channels, 2) + b) / gamma))
        k_size = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, 
                              padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_pool = self.avg_pool(x)
        x_squeezed = x_pool.squeeze(-1).unsqueeze(1)
        x_conv = self.conv(x_squeezed)
        x_weights = self.sigmoid(x_conv.squeeze(1).unsqueeze(-1))
        return x * x_weights

# ---------------------------------
# 模块 0.2: SimAM (无变化)
# ---------------------------------
class SimAmAttention1D(nn.Module):
    """ 1D 适配版 SimAM (无参数) """
    def __init__(self, e_lambda=1e-4):
        super(SimAmAttention1D, self).__init__()
        self.e_lambda = e_lambda

    def forward(self, x):
        if x.shape[2] <= 1:
            return x
        mu = torch.mean(x, dim=2, keepdim=True)
        sigma_sq = torch.mean((x - mu) ** 2, dim=2, keepdim=True)
        e = (x - mu) ** 2 / (sigma_sq + self.e_lambda) + 1
        weights = torch.sigmoid(1 / e)
        return x * weights

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

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out

# ---------------------------------
# ### 新增 ### 模块 1.2-SW-ECA-SimAM: B+方案 (BasicBlock 版本)
# ---------------------------------
class SW_ECA_SimAM_BasicBlock1D(nn.Module):
    """ 
    BasicBlock1D (B+方案):
    使用 3x3 SWConv, 并在残差相加前应用 ECA -> SimAM
    """
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, groups=1, **kwargs):
        super(SW_ECA_SimAM_BasicBlock1D, self).__init__()
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
        
        # ### 新增 ###: 实例化两个注意力模块 (通道数为 out_channel)
        self.eca = EcaChannelAttention(out_channel)
        self.simam = SimAmAttention1D()
        
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
        
        # ### 新 B+ 方案 ###: 依次应用 ECA 和 SimAM
        out = self.eca(out)
        out = self.simam(out)

        out += identity
        out = self.relu(out)
        return out

# ---------------------------------
# (为保持完整性，保留 Bottleneck 模块)
# ---------------------------------
class SW_Bottleneck1D(nn.Module):
    expansion = 4 
    def __init__(self, in_channel, out_channel, stride=1, downsample=None, groups=1, width_per_group=64, **kwargs):
        super(SW_Bottleneck1D, self).__init__()
        n_div = kwargs.get('n_div', 4)
        width = int(out_channel * (width_per_group / 64.)) * groups
        self.conv1 = nn.Conv1d(in_channels=in_channel, out_channels=width, kernel_size=1, stride=1, bias=False, groups=groups)
        self.bn1 = nn.BatchNorm1d(width); self.relu = nn.ReLU(inplace=True)
        self.conv2 = ShiftwiseConv1d(in_channels=width, out_channels=width, kernel_size=3, stride=stride, padding=1, bias=False, groups=groups, n_div=n_div)
        self.bn2 = nn.BatchNorm1d(width)
        self.conv3 = nn.Conv1d(in_channels=width, out_channels=out_channel * self.expansion, kernel_size=1, stride=1, bias=False, groups=groups)
        self.bn3 = nn.BatchNorm1d(out_channel * self.expansion); self.downsample = downsample
    def forward(self, x):
        identity = x;
        if self.downsample is not None: identity = self.downsample(x)
        out = self.conv1(x); out = self.bn1(out); out = self.relu(out)
        out = self.conv2(out); out = self.bn2(out); out = self.relu(out)
        out = self.conv3(out); out = self.bn3(out);
        out += identity; out = self.relu(out); return out

class SW_ECA_Bottleneck1D(nn.Module):
    expansion = 4 
    def __init__(self, in_channel, out_channel, stride=1, downsample=None, groups=1, width_per_group=64, **kwargs):
        super(SW_ECA_Bottleneck1D, self).__init__()
        n_div = kwargs.get('n_div', 4); width = int(out_channel * (width_per_group / 64.)) * groups
        self.conv1 = nn.Conv1d(in_channels=in_channel, out_channels=width, kernel_size=1, stride=1, bias=False, groups=groups)
        self.bn1 = nn.BatchNorm1d(width); self.relu = nn.ReLU(inplace=True)
        self.conv2 = ShiftwiseConv1d(in_channels=width, out_channels=width, kernel_size=3, stride=stride, padding=1, bias=False, groups=groups, n_div=n_div)
        self.bn2 = nn.BatchNorm1d(width)
        self.conv3 = nn.Conv1d(in_channels=width, out_channels=out_channel * self.expansion, kernel_size=1, stride=1, bias=False, groups=groups)
        self.bn3 = nn.BatchNorm1d(out_channel * self.expansion); self.eca = EcaChannelAttention(out_channel * self.expansion); self.downsample = downsample
    def forward(self, x):
        identity = x;
        if self.downsample is not None: identity = self.downsample(x)
        out = self.conv1(x); out = self.bn1(out); out = self.relu(out)
        out = self.conv2(out); out = self.bn2(out); out = self.relu(out)
        out = self.conv3(out); out = self.bn3(out); out = self.eca(out);
        out += identity; out = self.relu(out); return out

class SW_ECA_SimAM_Bottleneck1D(nn.Module):
    expansion = 4 
    def __init__(self, in_channel, out_channel, stride=1, downsample=None, groups=1, width_per_group=64, **kwargs):
        super(SW_ECA_SimAM_Bottleneck1D, self).__init__()
        n_div = kwargs.get('n_div', 4); width = int(out_channel * (width_per_group / 64.)) * groups
        self.conv1 = nn.Conv1d(in_channels=in_channel, out_channels=width, kernel_size=1, stride=1, bias=False, groups=groups)
        self.bn1 = nn.BatchNorm1d(width); self.relu = nn.ReLU(inplace=True)
        self.conv2 = ShiftwiseConv1d(in_channels=width, out_channels=width, kernel_size=3, stride=stride, padding=1, bias=False, groups=groups, n_div=n_div)
        self.bn2 = nn.BatchNorm1d(width)
        self.conv3 = nn.Conv1d(in_channels=width, out_channels=out_channel * self.expansion, kernel_size=1, stride=1, bias=False, groups=groups)
        self.bn3 = nn.BatchNorm1d(out_channel * self.expansion); self.eca = EcaChannelAttention(out_channel * self.expansion); self.simam = SimAmAttention1D(); self.downsample = downsample
    def forward(self, x):
        identity = x;
        if self.downsample is not None: identity = self.downsample(x)
        out = self.conv1(x); out = self.bn1(out); out = self.relu(out)
        out = self.conv2(out); out = self.bn2(out); out = self.relu(out)
        out = self.conv3(out); out = self.bn3(out);
        out = self.eca(out); out = self.simam(out);
        out += identity; out = self.relu(out); return out

# ---------------------------------
# 模块 3: ResNet1D (无变化)
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

    def forward(self, x):
        if x.dim() == 4:
            x = x.mean(dim=-1)
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

# --- 原始 SWConv 工厂 (Base) ---
def SW_Res_Inc_Net_base(num_classes: int = 1000, include_top: bool = True, n_div: int = 4):
    """ResNet-18 结构 (SW_BasicBlock1D, [2,2,2,2])"""
    return Res_Inc_Net(SW_BasicBlock1D, num_classes=num_classes, include_top=include_top,
        layers=(2, 2, 2, 2), base_channels=64, dropout=0.5, n_div=n_div)

# ---------------------------------
# ### 新增 ### 模块 5: B+方案 (Base 版本) 的工厂函数
# ---------------------------------
def SW_ECA_SimAM_Res_Inc_Net_base(num_classes: int = 1000, include_top: bool = True, n_div: int = 4):
    """ResNet-18 结构 (SW_ECA_SimAM_BasicBlock1D, [2,2,2,2])"""
    return Res_Inc_Net(SW_ECA_SimAM_BasicBlock1D, num_classes=num_classes, include_top=include_top,
        layers=(2, 2, 2, 2), base_channels=64, dropout=0.5, n_div=n_div)

# --- (为保持完整性，保留 Bottleneck 工厂) ---
def SW_Res_Inc_Net_50(num_classes: int = 1000, include_top: bool = True, n_div: int = 4):
    return Res_Inc_Net(SW_Bottleneck1D, num_classes=num_classes, include_top=include_top,
        layers=(3, 4, 6, 3), base_channels=64, dropout=0.5, n_div=n_div)
def SW_ECA_Res_Inc_Net_50(num_classes: int = 1000, include_top: bool = True, n_div: int = 4):
    return Res_Inc_Net(SW_ECA_Bottleneck1D, num_classes=num_classes, include_top=include_top,
        layers=(3, 4, 6, 3), base_channels=64, dropout=0.5, n_div=n_div)
def SW_ECA_SimAM_Res_Inc_Net_50(num_classes: int = 1000, include_top: bool = True, n_div: int = 4):
    return Res_Net(SW_ECA_SimAM_Bottleneck1D, num_classes=num_classes, include_top=include_top,
        layers=(3, 4, 6, 3), base_channels=64, dropout=0.5, n_div=n_div)


# ---------------------------------
# ### 测试代码 (只关注 Base 版本) ###
# ---------------------------------
from torchsummary import summary
from thop import profile
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# (Batch, Channels, Height, Width) 或 (B, C, F, T)
inputs = torch.randn(1, 1, 512, 512).to(device) 

# --- 基线 (仅 SW) ---
print("\n--- [基线] SW-Conv ResNet-Base (SW_BasicBlock1D) ---")
model_sw_base = SW_Res_Inc_Net_base(num_classes=10, include_top=True, n_div=4).to(device)
flops_sw_base, params_sw_base = profile(model_sw_base, (inputs,), verbose=False)
print(f'Flops: {flops_sw_base/1e9:.2f} G, Params: {params_sw_base/1e6:.2f} M')
print(f'  (输入 {tuple(inputs.shape)} -> .mean(dim=-1) -> {tuple(inputs.mean(dim=-1).shape)})')


# --- B+方案 (SW + ECA + SimAM) ---
print("\n--- [B+方案] SW-Conv + ECA + SimAM ResNet-Base (SW_ECA_SimAM_BasicBlock1D) ---")
model_sw_eca_simam_base = SW_ECA_SimAM_Res_Inc_Net_base(num_classes=10, include_top=True, n_div=4).to(device)
flops_sw_eca_simam_base, params_sw_eca_simam_base = profile(model_sw_eca_simam_base, (inputs,), verbose=False)
print(f'Flops: {flops_sw_eca_simam_base/1e9:.2f} G, Params: {params_sw_eca_simam_base/1e6:.2f} M')