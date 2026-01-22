import torch.nn as nn
import torch
from torchsummary import summary
from thop import profile
import math

class SELayer1D(nn.Module):
    def __init__(self, in_channel, reduction=16):
        """
        1D 版本的 Squeeze-and-Excitation 模块
        :param in_channel: 输入通道数
        :param reduction: 缩减因子，用于控制 SE 模块的参数量
        """
        super(SELayer1D, self).__init__()
        # Squeeze 操作：全局平均池化
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # Excitation 操作：两个全连接层
        self.fc = nn.Sequential(
            nn.Linear(in_channel, in_channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channel // reduction, in_channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x shape: (B, C, L)
        b, c, _ = x.size()
        
        # Squeeze: (B, C, L) -> (B, C, 1) -> (B, C)
        y = self.avg_pool(x).view(b, c)
        
        # Excitation: (B, C) -> (B, C)
        y = self.fc(y)
        
        # Rescale: (B, C) -> (B, C, 1)
        y = y.view(b, c, 1)
        
        # (B, C, L) * (B, C, 1) -> (B, C, L)
        return x * y

# ---------------------------------
# 模块 1: 你的原始 BasicBlock1D (保留对比)
# ---------------------------------
class BasicBlock1D(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, groups=1, 
                 width_per_group=64,  # width_per_group 在此未用, 但保留签名一致
                 dilation_levels=None # 保留签名一致
                 ):
        super(BasicBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False, groups=groups)
        self.bn1 = nn.BatchNorm1d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
        self.bn2 = nn.BatchNorm1d(out_channel)
        self.downsample = downsample
        self.SELayer1D = SELayer1D(out_channel, reduction=16)

        self.bn3 = nn.BatchNorm1d(out_channel)

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.SELayer1D(out)
        out = self.bn3(out)
        out += identity
        out = self.relu(out)

        return out

class ResNet1D(nn.Module):

    def __init__(self,
                 block, # block 将是 MultiDilationBasicBlock1D
                 blocks_num,
                 num_classes=1000,
                 include_top=True,
                 groups=1,
                 width_per_group=64,
                 base_channels=64,
                 width_mult: float = 1.0,
                 stem_channels: int = None,
                 dropout: float = 0.0,
                 dilation_levels: list = None # 新增：传递给 block
                 ):
        super(ResNet1D, self).__init__()
        self.include_top = include_top
        self.blocks_num = list(blocks_num)
        self.in_channel = stem_channels if stem_channels is not None else base_channels
        self.groups = groups
        self.width_per_group = width_per_group
        self.base_channels = base_channels
        self.width_mult = float(width_mult)
        self.dropout_p = float(dropout)
        self.dilation_levels = dilation_levels if dilation_levels is not None else [1, 2, 4, 8]

        # 频谱图输入通道为1
        self.conv1 = nn.Conv1d(1, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        def _make_divisible(v: int, divisor: int) -> int:
            if divisor <= 1:
                return int(v)
            v = int(v)
            # 确保通道数能被空洞分组整除
            divisor = max(divisor, len(self.dilation_levels)) 
            return max(divisor, (v + divisor - 1) // divisor * divisor)

        stage_base = [1, 2, 4, 8]
        stage_channels = [
            _make_divisible(int(self.base_channels * b * self.width_mult), self.groups)
            for b in stage_base
        ]
        
        # 确保 stage_channels 也能被 dilation_levels 的数量整除
        stage_channels = [
             _make_divisible(c, len(self.dilation_levels)) for c in stage_channels
        ]
        
        
        self.layer1 = self._make_layer(block, stage_channels[0], self.blocks_num[0])
        self.layer2 = self._make_layer(block, stage_channels[1], self.blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, stage_channels[2], self.blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, stage_channels[3], self.blocks_num[3], stride=2)

        self.dropout = nn.Dropout(p=self.dropout_p) if self.dropout_p and include_top else nn.Identity()
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool1d(1)
            # 使用最后一个存在 block 的 stage 的通道数，支持 (x, x, 0, 0) 这类两层网络
            last_idx = 0
            for i, b in enumerate(self.blocks_num):
                if b and b > 0:
                    last_idx = i
            final_channels = stage_channels[last_idx] * block.expansion
            self.fc = nn.Linear(final_channels, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

        self.thresholds = nn.Parameter(torch.ones(num_classes) * 0.5)

    def _make_layer(self, block, channel, block_num, stride=1):
        # 当希望该 stage 跳过（block_num==0）时，直接返回恒等映射
        if block_num <= 0:
            return nn.Identity()

        downsample = None
        
        # 确保 self.in_channel 也能被整除
        div = len(self.dilation_levels)
        if self.in_channel % div != 0:
            self.in_channel = math.ceil(self.in_channel / div) * div

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
                            dilation_levels=self.dilation_levels # 传递空洞率
                            ))
        
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channel,
                                channel,
                                groups=self.groups,
                                width_per_group=self.width_per_group,
                                dilation_levels=self.dilation_levels # 传递空洞率
                                ))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.mean(dim=-1)

        # 假设 x 的输入形状是 (B, 1, L)
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
# 模块 5: 工厂函数 (全部更新为使用新模块)
# ---------------------------------
def Res_Inc_Net(
    block,  # <--- 传入 MultiDilationBasicBlock1D
    num_classes: int = 1000,
    include_top: bool = True,
    layers: tuple[int, int, int, int] = (1, 1, 1, 1),
    groups: int = 32, # ResNeXt-like
    base_channels: int = 64,
    width_mult: float = 1.0,
    stem_channels: int = None,
    dropout: float = 0.5,
    dilation_levels: list = [1, 2, 4, 8] # 定义空洞率
):
    # 确保 base_channels 能被 dilation_levels 数量整除
    div = len(dilation_levels)
    if base_channels % div != 0:
        base_channels = math.ceil(base_channels / div) * div
        print(f"Warning: base_channels 调整为 {base_channels} 以匹配 {div} 个空洞率路径。")
        
    if stem_channels is not None and stem_channels % div != 0:
        stem_channels = math.ceil(stem_channels / div) * div

    return ResNet1D(
        block,  # <-- 传入 block
        list(layers),
        num_classes=num_classes,
        include_top=include_top,
        groups=groups,
        width_per_group=4, # 这个参数在 BasicBlock 中未用, 但在 Bottleneck 中有用
        base_channels=base_channels,
        width_mult=width_mult,
        stem_channels=stem_channels,
        dropout=dropout,
        dilation_levels=dilation_levels
    )

def Res_Inc_Net_nano(num_classes: int = 1000, include_top: bool = True):
    # 16 / 4 = 4. base_channels=16 也能被 4 整除
    return Res_Inc_Net(BasicBlock1D, num_classes=num_classes, include_top=include_top,
        layers=(1, 1, 1, 1), base_channels=8, groups=8, dropout=0.1)

def Res_Inc_Net_mini(num_classes: int = 1000, include_top: bool = True):
    # 16 / 4 = 4. base_channels=16 也能被 4 整除
    return Res_Inc_Net(BasicBlock1D, num_classes=num_classes, include_top=include_top,
        layers=(1, 1, 1, 1), base_channels=16, groups=16, dropout=0.2)
        
def Res_Inc_Net_tiny(num_classes: int = 1000, include_top: bool = True):
    # 32 / 4 = 8. base_channels=32 也能被 4 整除
    return Res_Inc_Net(BasicBlock1D, num_classes=num_classes, include_top=include_top,
        layers=(1, 1, 1, 1), base_channels=32, groups=32, dropout=0.3)

def Res_Inc_Net_small(num_classes: int = 1000, include_top: bool = True):
    # 64 / 4 = 16. base_channels=64 也能被 4 整除
    return Res_Inc_Net(BasicBlock1D, num_classes=num_classes, include_top=include_top,
        layers=(1, 1, 1, 1), base_channels=64, groups=32, dropout=0.5)

def Res_Inc_Net_base(num_classes: int = 1000, include_top: bool = True):
    return Res_Inc_Net(BasicBlock1D, num_classes=num_classes, include_top=include_top,
        layers=(2, 2, 2, 2), base_channels=64, groups=32, dropout=0.5)

def Res_Inc_Net_large(num_classes: int = 1000, include_top: bool = True):
    return Res_Inc_Net(BasicBlock1D, num_classes=num_classes, include_top=include_top,
        layers=(3, 4, 6, 3), base_channels=64, groups=32, dropout=0.5)

# ---------------------------------
# 模块 6: 测试
# ---------------------------------
if __name__ == "__main__":
    from torchsummary import summary
    from thop import profile
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for model_func, name in [
        (Res_Inc_Net_nano, "Res_Inc_Net_nano"),
        (Res_Inc_Net_mini, "Res_Inc_Net_mini"),
        (Res_Inc_Net_tiny, "Res_Inc_Net_tiny"),
        (Res_Inc_Net_small, "Res_Inc_Net_small"),
        (Res_Inc_Net_base, "Res_Inc_Net_base"),
        (Res_Inc_Net_large, "Res_Inc_Net_large")
    ]:
        print(f"--- Testing {name} ---")
        model = model_func(num_classes=10, include_top=True)
        model = model.to(device)
        inputs = torch.randn(1, 1, 512, 512).to(device) # 1D 频谱向量
        flops, params = profile(model, (inputs,), verbose=False)
        print(name, 'flops: ', flops, 'params: ', params)