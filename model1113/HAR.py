import torch.nn as nn
import torch
from torchsummary import summary
from thop import profile
import math

# ---------------------------------
# 模块 1: BasicBlock1D (原始)
# ---------------------------------
class BasicBlock1D(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, groups=1, width_per_group=64, 
                 dilation: int = 1, 
                 **kwargs): 
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
# 模块 2: Bottleneck1D (原始)
# ---------------------------------
class Bottleneck1D(nn.Module):
    expansion = 4 

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, groups=1, width_per_group=64, 
                 dilation: int = 1,
                 **kwargs):
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
# 模块 3: HAR_Block (新增的核心模块)
# ---------------------------------
class HAR_Block(nn.Module):
    expansion = 1

    def __init__(self, 
                 in_channel, 
                 out_channel, 
                 stride=1, 
                 downsample=None, 
                 groups=1, 
                 width_per_group=64,
                 dilation: int = 4, 
                 reduction: int = 16):
        
        super(HAR_Block, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        
        # 路径 A: 局部
        self.conv_local = nn.Conv1d(in_channels=in_channel, out_channels=out_channel,
                                    kernel_size=3, stride=stride, padding=1, 
                                    dilation=1, groups=groups, bias=False)
        self.bn_local = nn.BatchNorm1d(out_channel)
        
        # 路径 B: 周期性
        periodic_padding = (3 - 1) * dilation // 2
        self.conv_periodic = nn.Conv1d(in_channels=in_channel, out_channels=out_channel,
                                       kernel_size=3, stride=stride, padding=periodic_padding, 
                                       dilation=dilation, groups=groups, bias=False)
        self.bn_periodic = nn.BatchNorm1d(out_channel)
        
        # 融合 C: 注意力
        self.gap = nn.AdaptiveAvgPool1d(1)
        squeeze_channels = max(1, out_channel // reduction)
        self.fc_squeeze = nn.Linear(out_channel, squeeze_channels)
        self.fc_excite = nn.Linear(squeeze_channels, 2)
        self.softmax = nn.Softmax(dim=1) 
        
        self.bn_out = nn.BatchNorm1d(out_channel)

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
            
        x_local = self.conv_local(x)
        x_local = self.bn_local(x_local)
        
        x_periodic = self.conv_periodic(x)
        x_periodic = self.bn_periodic(x_periodic)

        x_fused = self.relu(x_local + x_periodic)
        u = self.gap(x_fused).squeeze(-1) 
        z = self.fc_squeeze(u) 
        z = self.relu(z)
        weights = self.softmax(self.fc_excite(z)) 
        
        w_local = weights[:, 0].unsqueeze(-1).unsqueeze(-1)
        w_periodic = weights[:, 1].unsqueeze(-1).unsqueeze(-1)
        
        out = (x_local * w_local) + (x_periodic * w_periodic)
        out = self.bn_out(out)
        
        out += identity
        out = self.relu(out)

        return out

# ---------------------------------
# 模块 4: ResNet1D (*** 已更新 forward ***)
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
                 dilations_per_layer: dict = None):
        super(ResNet1D, self).__init__()
        self.include_top = include_top
        self.in_channel = stem_channels if stem_channels is not None else base_channels
        self.groups = groups
        self.width_per_group = width_per_group
        self.base_channels = base_channels
        self.width_mult = float(width_mult)
        self.dropout_p = float(dropout)

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
        
        if dilations_per_layer is None:
            dilations_per_layer = {}
        
        default_dilation = 1
        
        def get_dilations(stage_name, num):
            return dilations_per_layer.get(stage_name, [default_dilation] * num)

        dilations_l1 = get_dilations('layer1', blocks_num[0])
        dilations_l2 = get_dilations('layer2', blocks_num[1])
        dilations_l3 = get_dilations('layer3', blocks_num[2])
        dilations_l4 = get_dilations('layer4', blocks_num[3])
        
        self.layer1 = self._make_layer(block, stage_channels[0], blocks_num[0], dilations=dilations_l1)
        self.layer2 = self._make_layer(block, stage_channels[1], blocks_num[1], stride=2, dilations=dilations_l2)
        self.layer3 = self._make_layer(block, stage_channels[2], blocks_num[2], stride=2, dilations=dilations_l3)
        self.layer4 = self._make_layer(block, stage_channels[3], blocks_num[3], stride=2, dilations=dilations_l4)

        self.dropout = nn.Dropout(p=self.dropout_p) if self.dropout_p and include_top else nn.Identity()
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool1d(1)
            final_channels = stage_channels[3] * block.expansion
            self.fc = nn.Linear(final_channels, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

        self.thresholds = nn.Parameter(torch.ones(num_classes) * 0.5)

    def _make_layer(self, block, channel, block_num, stride=1, dilations: list[int] = None):
        downsample = None
        
        if dilations is None:
            dilations = [1] * block_num
        
        if len(dilations) != block_num:
            raise ValueError(f"dilations 列表长度 {len(dilations)} 必须等于 block_num {block_num}")

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
                            dilation=dilations[0]))
        
        self.in_channel = channel * block.expansion
        
        for i in range(1, block_num):
            layers.append(block(self.in_channel,
                                channel,
                                groups=self.groups,
                                width_per_group=self.width_per_group,
                                dilation=dilations[i]))

        return nn.Sequential(*layers)

    def forward(self, x):
        # *** 新增: 根据您的要求内置预处理 ***
        # 允许输入 (B, 1, 512, 512) 或 (B, 1, 512)
        if x.dim() == 4:
            x = x.mean(dim=-1) # (B, 1, 512, 512) -> (B, 1, 512)
        
        # --- 1D ResNet 流程开始 ---
        # x 现在的形状是 (B, 1, L)，例如 (B, 1, 512)
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
# 模块 5: 工厂函数 (与之前相同)
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
    dilations_per_layer: dict = None
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
        dilations_per_layer=dilations_per_layer
    )

# --- 原始工厂 (BasicBlock1D) ---
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

# --- 原始工厂 (Bottleneck) ---
def Res_Inc_Net_50(num_classes: int = 1000, include_top: bool = True):
    """ResNet-50 结构 (Bottleneck, [3,4,6,3])"""
    return Res_Inc_Net(Bottleneck1D, num_classes=num_classes, include_top=include_top,
        layers=(3, 4, 6, 3), base_channels=64, dropout=0.5)

# --- *** 你的新 HAR-Net 工厂 (HAR_Block) *** ---

def HAR_Net_tiny(num_classes: int = 1000, include_top: bool = True):
    """HAR-Net-Tiny 结构 (HAR_Block, [1,1,1,1], base=32)"""
    layers_config = (1, 1, 1, 1)
    dilation_schedule = {
        'layer1': [2], 'layer2': [4], 'layer3': [8], 'layer4': [16]
    }
    return Res_Inc_Net(
        HAR_Block, num_classes=num_classes, include_top=include_top,
        layers=layers_config, base_channels=32, dropout=0.3,
        dilations_per_layer=dilation_schedule
    )

def HAR_Net_small(num_classes: int = 1000, include_top: bool = True):
    """HAR-Net-Small 结构 (HAR_Block, [1,1,1,1], base=64)"""
    layers_config = (1, 1, 1, 1)
    dilation_schedule = {
        'layer1': [2], 'layer2': [4], 'layer3': [8], 'layer4': [16]
    }
    return Res_Inc_Net(
        HAR_Block, num_classes=num_classes, include_top=include_top,
        layers=layers_config, base_channels=64, dropout=0.5,
        dilations_per_layer=dilation_schedule
    )

def HAR_Net_base(num_classes: int = 1000, include_top: bool = True, dropout: float = 0.5):
    """HAR-Net-18 结构 (HAR_Block, [2,2,2,2])"""
    layers_config = (2, 2, 2, 2)
    dilation_schedule = {
        'layer1': [2, 4], 'layer2': [8, 16], 'layer3': [32, 2], 'layer4': [4, 8]
    }
    return Res_Inc_Net(
        HAR_Block, num_classes=num_classes, include_top=include_top,
        layers=layers_config, base_channels=64, dropout=dropout,
        dilations_per_layer=dilation_schedule
    )

def HAR_Net_large(num_classes: int = 1000, include_top: bool = True):
    """HAR-Net-34 结构 (HAR_Block, [3,4,6,3])"""
    layers_config = (3, 4, 6, 3) 
    dilation_schedule = {
        'layer1': [2, 4, 8], 'layer2': [2, 4, 8, 16],
        'layer3': [2, 4, 8, 16, 32, 8], 'layer4': [2, 4, 16]
    }
    return Res_Inc_Net(
        HAR_Block, num_classes=num_classes, include_top=include_top,
        layers=layers_config, base_channels=64, dropout=0.5,
        dilations_per_layer=dilation_schedule
    )



if __name__ == "__main__":
    # # ---------------------------------
    # # 模块 6: 测试与验证 (*** 已更新 ***)
    # # ---------------------------------
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # # *** 关键更新: 使用您指定的 (B, C, F, T) 输入 ***
    # input_tensor_2d = torch.randn(1, 1, 512, 512).to(device)
    # num_classes = 10 # 假设10分类

    # def compare_models(orig_factory, har_factory, name):
    #     """
    #     辅助函数，用于分析和比较一对模型。
    #     """
    #     print(f"\n--- 比较: {name} vs HAR-{name} ---")
        
    #     # 1. 原始模型 (BasicBlock1D)
    #     model_original = orig_factory(num_classes=num_classes).to(device)
    #     # 使用正确的 (1, 1, 512, 512) 输入进行分析
    #     flops_orig, params_orig = profile(model_original, (input_tensor_2d.clone(),), verbose=False)
    #     print(f"  {name} (BasicBlock1D):")
    #     print(f"    FLOPs: {flops_orig:,.0f} | Params: {params_orig:,.0f}")
        
    #     # 2. HAR 模型 (HAR_Block)
    #     model_har = har_factory(num_classes=num_classes).to(device)
    #     # 使用正确的 (1, 1, 512, 512) 输入进行分析
    #     flops_har, params_har = profile(model_har, (input_tensor_2d.clone(),), verbose=False)
    #     print(f"  HAR-{name} (HAR_Block):")
    #     print(f"    FLOPs: {flops_har:,.0f} | Params: {params_har:,.0f}")
        
    #     # 3. 比较
    #     print(f"  -> FLOPs 变化: {flops_har/flops_orig:1.2%} | Params 变化: {params_har/params_orig:1.2%}")
    #     print("-" * 40)

    # # --- 运行所有比较 ---
    # print(f"基准输入: (1, 1, 512, 512), 分类数: {num_classes}")
    # print("模型将自动执行 .mean(dim=-1) 转换为 (1, 1, 512) 的 1D 向量")

    # compare_models(Res_Inc_Net_tiny, HAR_Net_tiny, "Tiny")
    # compare_models(Res_Inc_Net_small, HAR_Net_small, "Small")
    # compare_models(Res_Inc_Net_base, HAR_Net_base, "Base")
    # compare_models(Res_Inc_Net_large, HAR_Net_large, "Large")

    # print("\n--- 可解释性 ---")
    # print("您可以通过 hook 任何 HAR_Block 的 forward 方法来提取 `weights` 变量 (B, 2)")
    # print("这将显示网络是更依赖 'Local' (权重0) 还是 'Periodic' (权重1) 路径。")

    # # # 示例: 检查 'Base' 模型的摘要
    # # print("\n--- HAR-Net-Base 摘要 ---")
    # # model_har_base = HAR_Net_base(num_classes=10).to(device)
    # # # 传入 (1, 1, 512, 512) 或 (1, 1, 512) 均可
    # # summary(model_har_base, input_size=(1, 512, 512))

    from torchsummary import summary
    from thop import profile
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = HAR_Net_small(num_classes=10, include_top=True)
    model = model.to(device)

    # # summary(model, input_size=(1, 512))
    # inputs = torch.randn(1, 1, 512, 512).to(device)
    # flops, params = profile(model, (inputs,), verbose=False)
    # print('flops: ', flops, 'params: ', params)
    
    model.load_state_dict(torch.load('/home/kk/lzk/SRSpecNet/1112/HAR_Net_small/model.pth', map_location=device))
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

    