import torch.nn as nn
import torch
from torchsummary import summary
from thop import profile
import math

class SELayer1D(nn.Module):
    def __init__(self, in_channel, reduction=16):
        """
        1D version of Squeeze-and-Excitation module
        :param in_channel: Input channel number
        :param reduction: Reduction factor, used to control the parameter amount of SE module
        """
        super(SELayer1D, self).__init__()
        # Squeeze operation: Global Average Pooling
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # Excitation operation: Two fully connected layers
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
# Module 1: Your original BasicBlock1D (kept for comparison)
# ---------------------------------
class BasicBlock1D(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, groups=1):
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

# ---------------------------------
# Module 4: ResNet1D (General)
# ---------------------------------
class ResNet1D(nn.Module):

    def __init__(self,
                 block, # block will be MultiDilationBasicBlock1D
                 blocks_num,
                 num_classes=1000,
                 include_top=True,
                 groups=1,
                 width_per_group=64,
                 base_channels=64,
                 width_mult: float = 1.0,
                 stem_channels: int = None,
                 dropout: float = 0.0,
                 dilation_levels: list = None # New: pass to block
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

        # Spectrogram input channel is 1
        self.conv1 = nn.Conv1d(1, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        def _make_divisible(v: int, divisor: int) -> int:
            if divisor <= 1:
                return int(v)
            v = int(v)
            # Ensure the number of channels can be divisible by the dilation groups
            divisor = max(divisor, len(self.dilation_levels)) 
            return max(divisor, (v + divisor - 1) // divisor * divisor)

        stage_base = [1, 2, 4, 8]
        stage_channels = [
            _make_divisible(int(self.base_channels * b * self.width_mult), self.groups)
            for b in stage_base
        ]
        
        # Ensure stage_channels can be divisible by the number of dilation_levels
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
            # Use the channel number of the last stage where block exists, support two-layer networks like (x, x, 0, 0)
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
        # Return identity mapping directly when we hope to skip this stage (block_num==0)
        if block_num <= 0:
            return nn.Identity()

        downsample = None
        
        # Ensure self.in_channel can also be divisible
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
                            dilation_levels=self.dilation_levels
                            ))
        
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channel,
                                channel,
                                groups=self.groups,
                                width_per_group=self.width_per_group,
                                dilation_levels=self.dilation_levels
                                ))

        return nn.Sequential(*layers)

    def forward(self, x):
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

def Res_Inc_Net(
    block,
    num_classes: int = 1000,
    include_top: bool = True,
    layers: tuple[int, int, int, int] = (1, 1, 1, 1),
    groups: int = 32, # ResNeXt-like
    base_channels: int = 64,
    width_mult: float = 1.0,
    stem_channels: int = None,
    dropout: float = 0.5,
    dilation_levels: list = [1, 2, 4, 8]
):
    div = len(dilation_levels)
    if base_channels % div != 0:
        base_channels = math.ceil(base_channels / div) * div
        print(f"Warning: base_channels 调整为 {base_channels} 以匹配 {div} 个空洞率路径。")
        
    if stem_channels is not None and stem_channels % div != 0:
        stem_channels = math.ceil(stem_channels / div) * div

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
        dilation_levels=dilation_levels
    )

def Res_Inc_Net_nano(num_classes: int = 1000, include_top: bool = True):
    # 16 / 4 = 4. base_channels=16 can also be divisible by 4
    return Res_Inc_Net(BasicBlock1D, num_classes=num_classes, include_top=include_top,
        layers=(1, 1, 1, 1), base_channels=8, groups=8, dropout=0.1)

def Res_Inc_Net_mini(num_classes: int = 1000, include_top: bool = True):
    # 16 / 4 = 4. base_channels=16 can also be divisible by 4
    return Res_Inc_Net(BasicBlock1D, num_classes=num_classes, include_top=include_top,
        layers=(1, 1, 1, 1), base_channels=16, groups=16, dropout=0.2)
        
def Res_Inc_Net_tiny(num_classes: int = 1000, include_top: bool = True):
    # 32 / 4 = 8. base_channels=32 can also be divisible by 4
    return Res_Inc_Net(BasicBlock1D, num_classes=num_classes, include_top=include_top,
        layers=(1, 1, 1, 1), base_channels=32, groups=32, dropout=0.3)

def Res_Inc_Net_small(num_classes: int = 1000, include_top: bool = True):
    # 64 / 4 = 16. base_channels=64 can also be divisible by 4
    return Res_Inc_Net(BasicBlock1D, num_classes=num_classes, include_top=include_top,
        layers=(1, 1, 1, 1), base_channels=64, groups=32, dropout=0.5)

def Res_Inc_Net_base(num_classes: int = 1000, include_top: bool = True):
    return Res_Inc_Net(BasicBlock1D, num_classes=num_classes, include_top=include_top,
        layers=(2, 2, 2, 2), base_channels=64, groups=32, dropout=0.5)

def Res_Inc_Net_large(num_classes: int = 1000, include_top: bool = True):
    return Res_Inc_Net(BasicBlock1D, num_classes=num_classes, include_top=include_top,
        layers=(3, 4, 6, 3), base_channels=64, groups=32, dropout=0.5)


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
        inputs = torch.randn(1, 1, 512, 512).to(device) # 1D spectrum vector
        flops, params = profile(model, (inputs,), verbose=False)
        print(name, 'flops: ', flops, 'params: ', params)
        