import torch.nn as nn
import torch
from torchsummary import summary
from thop import profile

# ---------------------------------
# 模块 1 & 2: BasicBlock1D, Bottleneck1D (与你提供的一致，此处省略)
# ... (粘贴你之前的 BasicBlock1D 和 Bottleneck1D 代码)
# ---------------------------------
class BasicBlock1D(nn.Module):
    expansion = 1
    def __init__(self, in_channel, out_channel, stride=1, downsample=None, groups=1, **kwargs):
        super(BasicBlock1D, self).__init__()
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

class Bottleneck1D(nn.Module):
    expansion = 4 
    def __init__(self, in_channel, out_channel, stride=1, downsample=None, groups=1, width_per_group=64, **kwargs):
        super(Bottleneck1D, self).__init__()
        width = int(out_channel * (width_per_group / 64.)) * groups
        self.conv1 = nn.Conv1d(in_channels=in_channel, out_channels=width,
                               kernel_size=1, stride=1, bias=False, groups=groups)
        self.bn1 = nn.BatchNorm1d(width)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(in_channels=width, out_channels=width,
                               kernel_size=3, stride=stride, padding=1, bias=False, groups=groups)
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

# ---------------------------------
# 模块 3: ResNet1D (融合你的所有新要求)
# ---------------------------------
class ResNet1D_Hybrid(nn.Module): # «--- 重命名以示区别

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
                 # «--- ADDED: 两个新滤波器的参数 ---»
                 filter_2d_ksize: tuple = (7, 7), # (K_f, K_t)
                 filter_1d_ksize: int = 7
                ): 
        super(ResNet1D_Hybrid, self).__init__()
        self.include_top = include_top
        self.in_channel = stem_channels if stem_channels is not None else base_channels
        self.groups = groups
        self.width_per_group = width_per_group
        self.base_channels = base_channels
        self.width_mult = float(width_mult)
        self.dropout_p = float(dropout)

        # «--- 1. 2D匹配滤波器 (应对要求 2) ---»
        # 使用 padding='same' 来保持维度, 简化后续处理
        # 注意: (K_f, K_t) 必须是奇数
        padding_2d_f = filter_2d_ksize[0] // 2
        padding_2d_t = filter_2d_ksize[1] // 2
        
        self.filter_bank_2d = nn.Sequential(
            nn.Conv2d(
                in_channels=1, 
                out_channels=num_classes, # 每个类别一个2D模板
                kernel_size=filter_2d_ksize, 
                padding=(padding_2d_f, padding_2d_t),
                bias=False
            ),
            nn.BatchNorm2d(num_classes),
            nn.ReLU(inplace=True)
        )
        
        # «--- 2. 1D细化滤波器 (应对要求 1 & 3) ---»
        padding_1d = filter_1d_ksize // 2
        
        self.filter_bank_1d_refinement = nn.Sequential(
            nn.Conv1d(
                in_channels=num_classes, 
                out_channels=num_classes, # 独立细化
                kernel_size=filter_1d_ksize,
                padding=padding_1d,
                groups=num_classes, # «--- 核心：分组卷积
                bias=False
            ),
            nn.BatchNorm1d(num_classes),
            nn.ReLU(inplace=True)
        )

        # «--- 3. ResNet 主干的 Stem ---»
        # 它的输入通道 = 1D细化滤波器的输出通道 = num_classes
        self.conv1 = nn.Conv1d(num_classes, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # ... (ResNet1D 的 _make_layer 和 stage_channels 代码不变) ...
        def _make_divisible(v: int, divisor: int) -> int:
            if divisor <= 1: return int(v)
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

        # ... (分类头和权重初始化代码不变) ...
        self.dropout = nn.Dropout(p=self.dropout_p) if self.dropout_p and include_top else nn.Identity()
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool1d(1)
            final_channels = stage_channels[3] * block.expansion
            self.fc = nn.Linear(final_channels, num_classes)
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d)): # «--- 增加了Conv2d
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        self.thresholds = nn.Parameter(torch.ones(num_classes) * 0.5)
        
    # _make_layer 方法和你提供的一致
    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channel, channel * block.expansion, kernel_size=1,
                          stride=stride, bias=False, groups=self.groups),
                nn.BatchNorm1d(channel * block.expansion))
        layers = []
        layers.append(block(self.in_channel, channel, downsample=downsample,
                            stride=stride, groups=self.groups,
                            width_per_group=self.width_per_group))
        self.in_channel = channel * block.expansion
        for _ in range(1, block_num):
            layers.append(block(self.in_channel, channel, groups=self.groups,
                                  width_per_group=self.width_per_group))
        return nn.Sequential(*layers)


    def forward(self, x):
        # 初始 x: (B, 1, F, T)，例如 (B, 1, 512, 512)
        
        # «--- 1. 2D 匹配 (应对要求 2) ---»
        # h1: (B, num_classes, F, T)
        h1 = self.filter_bank_2d(x) 
        
        # «--- 2. F轴压缩 (应对要求 2) ---»
        # h2: (B, num_classes, T)
        h2 = torch.mean(h1, dim=2) 
        
        # «--- 3. 1D 细化 (应对要求 1 & 3) ---»
        # h3: (B, num_classes, T)
        h3 = self.filter_bank_1d_refinement(h2)
        
        # «--- 4. ResNet Stem ---»
        # h3 现在是 (B, num_classes, T)
        h4 = self.conv1(h3) 
        h4 = self.bn1(h4)
        h4 = self.relu(h4)
        h4 = self.maxpool(h4) 

        # «--- 5. ResNet 主体 ---»
        h5 = self.layer1(h4)
        h5 = self.layer2(h5)
        h5 = self.layer3(h5)
        h5 = self.layer4(h5)

        # «--- 6. 分类头 ---»
        if self.include_top:
            out = self.avgpool(h5)
            out = torch.flatten(out, 1)
            out = self.dropout(out)
            out = self.fc(out)
        else:
            out = h5

        return out

# ---------------------------------
# 模块 4: 工厂函数 (已修改)
# ---------------------------------
# 通用工厂
def Res_Inc_Net_Hybrid(
    block,
    num_classes: int = 1000,
    include_top: bool = True,
    layers: tuple[int, int, int, int] = (1, 1, 1, 1),
    groups: int = 32,
    base_channels: int = 64,
    width_mult: float = 1.0,
    stem_channels: int = None,
    dropout: float = 0.5,
    **kwargs # «--- 使用 kwargs 传递滤波器参数
):
    return ResNet1D_Hybrid( # «--- 使用新模型
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
        **kwargs # «--- 传递 filter_2d_ksize, filter_1d_ksize
    )

# ResNet-18 (Hybrid)
def Res_Inc_Net_base_Hybrid(num_classes: int = 1000, include_top: bool = True, **kwargs):
    return Res_Inc_Net_Hybrid(BasicBlock1D, num_classes=num_classes, include_top=include_top,
                              layers=(2, 2, 2, 2), base_channels=64, dropout=0.5, **kwargs)

# ResNet-50 (Hybrid)
def Res_Inc_Net_50_Hybrid(num_classes: int = 1000, include_top: bool = True, **kwargs):
    return Res_Inc_Net_Hybrid(Bottleneck1D, num_classes=num_classes, include_top=include_top,
                              layers=(3, 4, 6, 3), base_channels=64, dropout=0.5, **kwargs)

# # ---------------------------------
# # 模块 5: 测试和分析
# # ---------------------------------
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# NUM_CLASSES = 10
# INPUT_T_LEN = 512 
# INPUT_F_LEN = 512

# # «--- 试验你的滤波器尺寸 ---»
# # 假设 ZC 序列在 F 轴上占 11 个bins, T 轴上占 15 个bins
# K_2D = (11, 15)
# # 1D 细化滤波器的长度
# K_1D = 9

# model = Res_Inc_Net_base_Hybrid(
#     num_classes=NUM_CLASSES, 
#     include_top=True, 
#     filter_2d_ksize=K_2D, # «--- 传递参数
#     filter_1d_ksize=K_1D  # «--- 传递参数
# )
# model = model.to(device)

# # --- 2. 使用 thop 计算 FLOPs 和参数量 ---
# inputs = torch.randn(1, 1, INPUT_F_LEN, INPUT_T_LEN).to(device)
# model.eval() 

# flops, params = profile(model, (inputs,))

# print("\n--- Model Profile (Hybrid 2D-1D) ---")
# print(f"模型: Res_Inc_Net_base_Hybrid")
# print(f"2D 滤波器 K: {K_2D}, 1D 滤波器 K: {K_1D}")
# print(f"类别数: {NUM_CLASSES}")
# print(f"输入形状: {inputs.shape}")
# print(f'FLOPs: {flops / 1e9:.2f} G')
# print(f'Params: {params / 1e6:.2f} M')

# # --- 验证假设 ---
# print("\n--- 验证假设 ---")
# print(f"2D 滤波器权重 (filter_bank_2d): {model.filter_bank_2d[0].weight.shape}")
# print(f"1D 细化滤波器权重 (filter_bank_1d_refinement): {model.filter_bank_1d_refinement[0].weight.shape}")
# print(f"ResNet Stem (conv1) 权重: {model.conv1.weight.shape}")