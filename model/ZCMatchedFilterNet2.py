import torch.nn as nn
import torch
from torchsummary import summary
from thop import profile

# ---------------------------------
# 模块 1: BasicBlock1D (来自你的代码)
# ---------------------------------
class BasicBlock1D(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, groups=1, **kwargs):
        super(BasicBlock1D, self).__init__()
        # 注意：kwargs 中的 width_per_group 在这里没有被使用
        self.conv1 = nn.Conv1d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False, groups=groups)
        self.bn1 = nn.BatchNorm1d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
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
# 模块 2: Bottleneck1D (来自你的代码)
# ---------------------------------
class Bottleneck1D(nn.Module):
    expansion = 4  # 瓶颈块的输出通道是中间层的4倍

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, groups=1, width_per_group=64, **kwargs):
        super(Bottleneck1D, self).__init__()
        
        # out_channel 在这里指 "瓶颈" 的通道数，例如 64
        # 最终输出通道将是 out_channel * self.expansion, 例如 256
        width = int(out_channel * (width_per_group / 64.)) * groups
        
        # 1x1 卷积：降维
        self.conv1 = nn.Conv1d(in_channels=in_channel, out_channels=width,
                               kernel_size=1, stride=1, bias=False, groups=groups)
        self.bn1 = nn.BatchNorm1d(width)
        self.relu = nn.ReLU(inplace=True)
        
        # 3x3 卷积：特征提取 (使用 group 卷积)
        self.conv2 = nn.Conv1d(in_channels=width, out_channels=width,
                               kernel_size=3, stride=stride, padding=1, bias=False, groups=groups)
        self.bn2 = nn.BatchNorm1d(width)
        
        # 1x1 卷积：升维
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
# 模块 3: ResNet1D (已集成你的设想)
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
                 init_filter_len: int = 7): # «--- ADDED: 匹配滤波器的长度
        super(ResNet1D, self).__init__()
        self.include_top = include_top
        self.in_channel = stem_channels if stem_channels is not None else base_channels
        self.groups = groups
        self.width_per_group = width_per_group
        self.base_channels = base_channels
        self.width_mult = float(width_mult)
        self.dropout_p = float(dropout)

        # «--- ADDED START: 1. 可学习的“匹配滤波器”层 (你的核心设想) ---»
        # 这一层将 (B, 1, T) 映射到 (B, num_classes, T)
        padding = init_filter_len // 2
        self.matched_filter_layer = nn.Sequential(
            nn.Conv1d(
                in_channels=1, 
                out_channels=num_classes, # 使用 num_classes 作为输出通道
                kernel_size=init_filter_len, 
                padding=padding,
                bias=False
            ),
            nn.BatchNorm1d(num_classes),
            nn.ReLU(inplace=True)
        )
        # «--- ADDED END ---»

        # «--- MODIFIED START: 2. ResNet 主干的 Stem ---»
        # 原来的 conv1 输入通道是 1
        # 现修改为 num_classes，以接收匹配滤波器的输出
        self.conv1 = nn.Conv1d(num_classes, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        # «--- MODIFIED END ---»
        
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
                            width_per_group=self.width_per_group))
        
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channel,
                                  channel,
                                  groups=self.groups,
                                  width_per_group=self.width_per_group))

        return nn.Sequential(*layers)

    def forward(self, x):
        # 初始 x: (B, 1, F, T)，例如 (1, 1, 512, 512)
        x = x.mean(dim=2) # F轴平均 -> (B, 1, T)
        
        # «--- ADDED: 1. 通过匹配滤波器层 ---»
        x = self.matched_filter_layer(x) # -> (B, num_classes, T)
        
        # «--- 2. 通过 ResNet Stem ---»
        # x 现在是 (B, num_classes, T)
        x = self.conv1(x) # -> (B, in_channel, T/2)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x) # -> (B, in_channel, T/4)

        # «--- 3. 通过 ResNet 主体 ---»
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # «--- 4. 通过分类头 ---»
        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.dropout(x)
            x = self.fc(x)

        return x

# ---------------------------------
# 模块 4: 工厂函数 (已修改以传递 init_filter_len)
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
    init_filter_len: int = 7 # «--- ADDED
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
        init_filter_len=init_filter_len # «--- ADDED: 传递参数
    )

# --- 你的原始工厂 (已修改) ---
def Res_Inc_Net_tiny(num_classes: int = 1000, include_top: bool = True, **kwargs): # «--- MODIFIED
    return Res_Inc_Net(BasicBlock1D, num_classes=num_classes, include_top=include_top,
                       layers=(1, 1, 1, 1), base_channels=32, dropout=0.3, **kwargs) # «--- MODIFIED

def Res_Inc_Net_small(num_classes: int = 1000, include_top: bool = True, **kwargs): # «--- MODIFIED
    return Res_Inc_Net(BasicBlock1D, num_classes=num_classes, include_top=include_top,
                       layers=(1, 1, 1, 1), base_channels=64, dropout=0.5, **kwargs) # «--- MODIFIED

def Res_Inc_Net_base(num_classes: int = 1000, include_top: bool = True, **kwargs): # «--- MODIFIED
    """ResNet-18 结构 (BasicBlock, [2,2,2,2])"""
    return Res_Inc_Net(BasicBlock1D, num_classes=num_classes, include_top=include_top,
                       layers=(2, 2, 2, 2), base_channels=64, dropout=0.5, **kwargs) # «--- MODIFIED

def Res_Inc_Net_large(num_classes: int = 1000, include_top: bool = True, **kwargs): # «--- MODIFIED
    """ResNet-34 结构 (BasicBlock, [3,4,6,3])"""
    return Res_Inc_Net(BasicBlock1D, num_classes=num_classes, include_top=include_top,
                       layers=(3, 4, 6, 3), base_channels=64, dropout=0.5, **kwargs) # «--- MODIFIED

# --- 更强的新工厂 (已修改) ---
def Res_Inc_Net_50(num_classes: int = 1000, include_top: bool = True, **kwargs): # «--- MODIFIED
    """ResNet-50 结构 (Bottleneck, [3,4,6,3])"""
    return Res_Inc_Net(Bottleneck1D, num_classes=num_classes, include_top=include_top,
                       layers=(3, 4, 6, 3), base_channels=64, dropout=0.5, **kwargs) # «--- MODIFIED

def Res_Inc_Net_101(num_classes: int = 1000, include_top: bool = True, **kwargs): # «--- MODIFIED
    """ResNet-101 结构 (Bottleneck, [3,4,23,3])"""
    return Res_Inc_Net(Bottleneck1D, num_classes=num_classes, include_top=include_top,
                       layers=(3, 4, 23, 3), base_channels=64, dropout=0.5, **kwargs) # «--- MODIFIED

# # ---------------------------------
# # 模块 5: 测试和分析
# # ---------------------------------
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# # 假设你的分类任务有 10 个类别
# NUM_CLASSES = 10
# # 假设你的输入频谱图时间轴长度为 512
# INPUT_T_LEN = 512 
# INPUT_F_LEN = 512

# # 你可以调整匹配滤波器的长度
# FILTER_LEN = 11 

# model = Res_Inc_Net_base(
#     num_classes=NUM_CLASSES, 
#     include_top=True, 
#     init_filter_len=FILTER_LEN # «--- 传递自定义滤波器长度
# )
# model = model.to(device)

# # --- 1. 使用 torchsummary 查看维度变化 ---
# # 注意: summary 需要的 input_size 是 (C, H, W) 或 (C, L)
# # 我们的 forward 接受 (1, F, T)，但内部处理 (1, T)
# # 所以我们给 summary 一个它能理解的、模拟 F 轴平均后的输入
# print("--- Model Summary (模拟 F 轴平均后) ---")
# # summary(model, input_size=(1, INPUT_F_LEN, INPUT_T_LEN)) # thop 更准确
# # summary 库可能无法正确处理 .mean() 操作，我们专注于 thop

# # --- 2. 使用 thop 计算 FLOPs 和参数量 ---
# # thop 需要一个示例输入张量
# # 我们的模型 forward 函数会自己处理 (B, C, F, T) -> (B, C, T)
# inputs = torch.randn(1, 1, 512, 512).to(device)

# # 确保模型在 eval 模式下进行 profile
# model.eval() 
# flops, params = profile(model, (inputs,))

# print("\n--- Model Profile (thop) ---")
# print(f"模型: Res_Inc_Net_base (集成了 {FILTER_LEN}-len 匹配滤波器)")
# print(f"类别数: {NUM_CLASSES}")
# print(f"输入形状: {inputs.shape}")
# print(f'FLOPs: {flops / 1e9:.2f} G')
# print(f'Params: {params / 1e6:.2f} M')