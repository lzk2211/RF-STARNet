import torch.nn as nn
import torch
from torchsummary import summary
from thop import profile

# ---------------------------------
# 模块 1: BasicBlock1D (你的原始代码)
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
# 模块 2: Bottleneck1D (更强的新模块)
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
# 模块 3: ResNet1D (已修正 forward)
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
                 dropout: float = 0.0):
        super(ResNet1D, self).__init__()
        self.include_top = include_top
        self.in_channel = stem_channels if stem_channels is not None else base_channels
        self.groups = groups
        self.width_per_group = width_per_group
        self.base_channels = base_channels
        self.width_mult = float(width_mult)
        self.dropout_p = float(dropout)

        # 修改输入通道数为1
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
        
        # 对于 Bottleneck，stage_channels[0] (例如64) 是 *瓶颈* 通道数
        # BasicBlock 会将其视为 *输出* 通道数
        # _make_layer 中的逻辑会正确处理
        
        self.layer1 = self._make_layer(block, stage_channels[0], blocks_num[0])
        self.layer2 = self._make_layer(block, stage_channels[1], blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, stage_channels[2], blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, stage_channels[3], blocks_num[3], stride=2)

        self.dropout = nn.Dropout(p=self.dropout_p) if self.dropout_p and include_top else nn.Identity()
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool1d(1)
            # 最终通道数是最后一个 stage 的通道数 * 块的 expansion
            final_channels = stage_channels[3] * block.expansion
            self.fc = nn.Linear(final_channels, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

        self.thresholds = nn.Parameter(torch.ones(num_classes) * 0.5)

    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        # 'channel' 是传入的 stage_channel，例如 64, 128, 256, 512
        # block.expansion 对 BasicBlock 是 1, 对 Bottleneck 是 4
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channel, channel * block.expansion, kernel_size=1,
                          stride=stride, bias=False, groups=self.groups),
                nn.BatchNorm1d(channel * block.expansion))

        layers = []
        layers.append(block(self.in_channel,
                            channel,  # 'channel' 传递给 block 的 out_channel
                            downsample=downsample,
                            stride=stride,
                            groups=self.groups,
                            width_per_group=self.width_per_group))
        
        # 更新 self.in_channel 供下一块使用
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channel,
                                channel,
                                groups=self.groups,
                                width_per_group=self.width_per_group))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.mean(dim=-1)  # <-- 已删除此行错误代码
        
        # 假设 x 的输入形状是 (B, 1, L)，例如 (1, 1, 512)
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
# 模块 4: 工厂函数 (你的原始函数 + 新函数)
# ---------------------------------
def Res_Inc_Net(
    block,  # <--- 让通用工厂也接受 block
    num_classes: int = 1000,
    include_top: bool = True,
    layers: tuple[int, int, int, int] = (1, 1, 1, 1),
    groups: int = 32,
    base_channels: int = 64,
    width_mult: float = 1.0,
    stem_channels: int = None,
    dropout: float = 0.5,
):
    return ResNet1D(
        block,  # <-- 传入 block
        list(layers),
        num_classes=num_classes,
        include_top=include_top,
        groups=groups,
        width_per_group=4, # 这个参数在你的 BasicBlock 中未用，但在 Bottleneck 中有用
        base_channels=base_channels,
        width_mult=width_mult,
        stem_channels=stem_channels,
        dropout=dropout,
    )

# --- 你的原始工厂 ---
def Res_Inc_Net_tiny(num_classes: int = 1000, include_top: bool = True):
    return Res_Inc_Net(BasicBlock1D, num_classes=num_classes, include_top=include_top,
        layers=(1, 1, 1, 1), base_channels=32, dropout=0.3)

def Res_Inc_Net_small(num_classes: int = 1000, include_top: bool = True):
    return Res_Inc_Net(BasicBlock1D, num_classes=num_classes, include_top=include_top,
        layers=(1, 1, 1, 1), base_channels=64, dropout=0.5)

def Res_Inc_Net_base(num_classes: int = 1000, include_top: bool = True, dropout: float = 0.5):
    """ResNet-18 结构 (BasicBlock, [2,2,2,2])"""
    return Res_Inc_Net(BasicBlock1D, num_classes=num_classes, include_top=include_top,
        layers=(2, 2, 2, 2), base_channels=64, dropout=dropout)

def Res_Inc_Net_large(num_classes: int = 1000, include_top: bool = True):
    """ResNet-34 结构 (BasicBlock, [3,4,6,3])"""
    return Res_Inc_Net(BasicBlock1D, num_classes=num_classes, include_top=include_top,
        layers=(3, 4, 6, 3), base_channels=64, dropout=0.5)

# --- 更强的新工厂 ---
def Res_Inc_Net_50(num_classes: int = 1000, include_top: bool = True):
    """ResNet-50 结构 (Bottleneck, [3,4,6,3])"""
    return Res_Inc_Net(Bottleneck1D, num_classes=num_classes, include_top=include_top,
        layers=(3, 4, 6, 3), base_channels=64, dropout=0.5)

def Res_Inc_Net_101(num_classes: int = 1000, include_top: bool = True):
    """ResNet-101 结构 (Bottleneck, [3,4,23,3])"""
    return Res_Inc_Net(Bottleneck1D, num_classes=num_classes, include_top=include_top,
        layers=(3, 4, 23, 3), base_channels=64, dropout=0.5)



if __name__ == "__main__":
    from torchsummary import summary
    from thop import profile
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = Res_Inc_Net_small(num_classes=10, include_top=True)
    model = model.to(device)

    # summary(model, input_size=(1, 512))
    # inputs = torch.randn(1, 1, 512, 512).to(device)
    # flops, params = profile(model, (inputs,), verbose=False)
    # print('flops: ', flops, 'params: ', params)
    # flops:  3536384.0 params:  134986.0
    # 1. 实例化你的模型
    # 假设你的模型是 RFformer，并且需要 num_classes=10
    # model = YourModelClass(num_classes=10) # 替换成你自己的模型类和参数

    # 2. 加载你训练好的 .pth 权重文件
    # 注意：这里假设你的 .pth 文件是 state_dict
    model.load_state_dict(torch.load('/home/kk/lzk/SRSpecNet/1112/Res_Inc_Net_small/model.pth', map_location=device))
    model.eval() # 切换到评估模式

    # 3. 准备一个符合你模型输入的示例张量（dummy input）
    # 根据你的记忆，你的输入是 (bs, 1, 512, 512)
    # 我们这里用 batch_size=1
    dummy_input = torch.randn(1, 1, 512, 512).to(device)

    # 4. 导出为 ONNX
    torch.onnx.export(model,               # 你要导出的模型
                    dummy_input,         # 一个示例输入
                    "model.onnx",        # 导出的文件名
                    input_names=['input'], # 可选：指定输入名
                    output_names=['output']) # 可选：指定输出名

    print("模型已成功导出为 model.onnx")