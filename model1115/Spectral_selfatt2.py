import torch.nn as nn
import torch
import torch.nn.functional as F  # 导入 F
from torchsummary import summary
from thop import profile
import math

# ============================================================
# 模块 0: 新增 NoiseGateConv1d (来自 S3R_light.py)
# ============================================================
class NoiseGateConv1d(nn.Module):
    """ 1D 版本的稳定 NoiseGate """
    def __init__(self):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        # 原始可学习参数（softplus 后总 >= 0）
        self.k_raw = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        # x shape: [B, C, L]
        # k 始终为正且可控
        k = F.softplus(self.k_raw)

        mean = self.avgpool(x)  # [B, C, 1]
        sigma = (x - mean).abs().mean(dim=2, keepdim=True) # [B, C, 1]

        # 稳定 gate：避免 1/sigma 形式导致爆炸
        gate = 1.0 / (1.0 + k * sigma)

        # 限制 gate 范围，避免关死
        gate = torch.clamp(gate, 0.3, 1.0)

        return x * gate

# ============================================================
# 模块 1: SELayer1D (保留)
# ============================================================
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
# 模块 3: 【创新模块】MultiDilationBasicBlock1D
# ---------------------------------
class MultiDilationBasicBlock1D(nn.Module):
    """
    创新的多尺度空洞卷积块 (Multi-Dilation Basic Block)
    它取代 BasicBlock1D，利用分组卷积实现多尺度感受野
    """
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, groups=1, 
                 width_per_group=64, # ResNeXt 结构需要
                 dilation_levels=[1, 2, 4, 8] # 默认的空洞率
                 ):
        super(MultiDilationBasicBlock1D, self).__init__()
        
        self.num_dilation_levels = len(dilation_levels)
        
        # 确保通道数和组数可以被空洞率的层级整除
        if in_channel % self.num_dilation_levels != 0 or out_channel % self.num_dilation_levels != 0:
            raise ValueError(f"in_channel ({in_channel}) 和 out_channel ({out_channel}) "
                             f"必须能被 num_dilation_levels ({self.num_dilation_levels}) 整除。")
        if groups % self.num_dilation_levels != 0:
            raise ValueError(f"groups ({groups}) 必须能被 num_dilation_levels ({self.num_dilation_levels}) 整除。")
            
        # 每个空洞路径的通道数和组数
        self.in_ch_path = in_channel // self.num_dilation_levels
        self.out_ch_path = out_channel // self.num_dilation_levels
        self.groups_per_path = groups // self.num_dilation_levels

        self.conv1_paths = nn.ModuleList()
        for dilation in dilation_levels:
            # 动态计算 padding 以保持 'same' 卷积
            # padding = (kernel_size - 1) * dilation / 2
            padding = dilation # k=3 时, (3-1)*d/2 = d
            self.conv1_paths.append(nn.Conv1d(in_channels=self.in_ch_path, 
                                              out_channels=self.out_ch_path,
                                              kernel_size=3, 
                                              stride=stride, # stride 应用在第一个卷积层
                                              padding=padding, 
                                              bias=False, 
                                              groups=self.groups_per_path, 
                                              dilation=dilation))

        self.bn1 = nn.BatchNorm1d(out_channel)
        self.relu = nn.ReLU(inplace=True)

        # conv2 作为一个 "特征融合器"，混合来自不同空洞路径的信息
        # 它仍然使用分组卷积来保持 ResNeXt 结构
        self.conv2 = nn.Conv1d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, 
                               stride=1, # stride 永远为 1
                               padding=1, 
                               bias=False, 
                               groups=groups) # 保持总组数
        self.bn2 = nn.BatchNorm1d(out_channel)
        self.downsample = downsample

        self.SELayer1D = SELayer1D(out_channel, reduction=16)
        self.ng1 = NoiseGateConv1d()
        self.ng2 = NoiseGateConv1d()

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        # --- 1. Split-Transform (多尺度路径) ---
        # 沿着通道维度将输入拆分
        x_split = torch.split(x, self.in_ch_path, dim=1)
        path_outs = []
        for i in range(self.num_dilation_levels):
            path_outs.append(self.conv1_paths[i](x_split[i]))
        
        # --- 2. Merge ---
        out = torch.cat(path_outs, dim=1)

        out = self.bn1(out)
        out = self.relu(out)
        

        # --- 3. 混合 (conv2 融合) ---
        out = self.conv2(out)
        out = self.bn2(out)

        out = self.SELayer1D(out)
        out += identity
        out = self.relu(out)

        return out


# ============================================================
# 模块 5: ResNet1D (通用) - 【已修改】
# ============================================================
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
        
        # 🔥【修改点 1】: 增加 NoiseGateConv1d
        self.ng1 = NoiseGateConv1d()
        
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
            
            # 使用最后一个存在 block 的 stage 的通道数
            last_idx = 0
            for i, b in enumerate(self.blocks_num):
                if b and b > 0:
                    last_idx = i
            # `self.in_channel` 此时也等于 final_channels
            final_channels = stage_channels[last_idx] * block.expansion
            
            # 🔥【修改点 2】: 增加轻量级 TransformerEncoder
            nhead = 4 # 借鉴 S3R
            # 确保 d_model (final_channels) 可以被 nhead 整除
            if final_channels == 0: # 容错
                final_channels = 1 
            if final_channels % nhead != 0:
                 # 寻找一个可以整除的 nhead
                for h in [2, 1]:
                    if final_channels % h == 0:
                        nhead = h
                        break
                if final_channels % nhead != 0: # 如果还是不行
                     print(f"Warning: final_channels ({final_channels}) 无法被 4/2/1 整除，Transformer 可能失败。")

            # 仅当有层时才创建 (final_channels > 0)
            transformer_layer = nn.TransformerEncoderLayer(
                d_model=final_channels,
                nhead=nhead,
                dim_feedforward=final_channels * 2, # 轻量级 FFN
                dropout=0.05, # 借鉴 S3R
                activation="gelu", # 借鉴 S3R
                batch_first=True # (B, L, C)
            )
            # 使用 1 层 Transformer 作为轻量级自注意力
            self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=1) 
            
            # 最后的 FC 层，输入维度不变
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
        # 假设 x 的输入形状是 (B, 1, 512, 512)
        # 🔥【修改点 3】: 降维，与 S3R_light 保持一致
        if x.dim() == 4:
            x = x.mean(dim=-1) # (B, 1, 512)

        # 1. Stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        # 🔥【修改点 4】: 应用 NoiseGate
        x = self.ng1(x) 
        
        x = self.maxpool(x)

        # 2. ResNet Layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x) # (B, C, L)

        if self.include_top:
            # 🔥【修改点 5】: 应用轻量级 Transformer
            # (B, C, L) -> (B, L, C) 以符合 TransformerEncoder (batch_first=True)
            x = x.transpose(1, 2)
            x = self.transformer(x)
            # (B, L, C) -> (B, C, L) 以符合 AvgPool1d
            x = x.transpose(1, 2)
            
            # 3. Head
            x = self.avgpool(x) # (B, C, 1)
            x = torch.flatten(x, 1) # (B, C)
            x = self.dropout(x)
            x = self.fc(x)

        return x

# ============================================================
# 模块 6: 工厂函数 (全部更新为使用新模块)
# ============================================================
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

# --- 使用新模块的工厂 ---
# 默认使用 MultiDilationBasicBlock1D

def Res_Inc_Net_mini(num_classes: int = 1000, include_top: bool = True):
    # 16 / 4 = 4. base_channels=16 也能被 4 整除
    return Res_Inc_Net(MultiDilationBasicBlock1D, num_classes=num_classes, include_top=include_top,
        layers=(1, 1, 0, 0), base_channels=64, groups=32, dropout=0.3)

def Res_Inc_Net_tiny(num_classes: int = 1000, include_top: bool = True):
    # 32 / 4 = 8. base_channels=32 也能被 4 整除
    return Res_Inc_Net(MultiDilationBasicBlock1D, num_classes=num_classes, include_top=include_top,
        layers=(1, 1, 1, 1), base_channels=32, groups=32, dropout=0.3)

def Res_Inc_Net_small(num_classes: int = 1000, include_top: bool = True):
    # 64 / 4 = 16. base_channels=64 也能被 4 整除
    return Res_Inc_Net(MultiDilationBasicBlock1D, num_classes=num_classes, include_top=include_top,
        layers=(1, 1, 1, 1), base_channels=64, groups=32, dropout=0.5)

def Res_Inc_Net_base(num_classes: int = 1000, include_top: bool = True, dropout: float = 0.5):
    """ResNet-18 结构 (MultiDilationBasicBlock1D, [2,2,2,2])"""
    return Res_Inc_Net(MultiDilationBasicBlock1D, num_classes=num_classes, include_top=include_top,
        layers=(2, 2, 2, 2), base_channels=64, groups=32, dropout=dropout)

def Res_Inc_Net_large(num_classes: int = 1000, include_top: bool = True):
    """ResNet-34 结构 (MultiDilationBasicBlock1D, [3,4,6,3])"""
    return Res_Inc_Net(MultiDilationBasicBlock1D, num_classes=num_classes, include_top=include_top,
        layers=(3, 4, 6, 3), base_channels=64, groups=32, dropout=0.5)


# ============================================================
# 模块 7: 测试
# ============================================================
# if __name__ == "__main__":
#     from torchsummary import summary
#     from thop import profile
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#     # 🔥 测试 Res_Inc_Net_tiny
#     model_func = Res_Inc_Net_tiny
#     name = "Res_Inc_Net_tiny (v_transformer+noisegate)"
    
#     print(f"--- Testing {name} ---")
#     model = model_func(num_classes=10, include_top=True)
#     model = model.to(device)
    
#     # 检查 Res_Inc_Net_tiny (layers=(1,1,1,1), base_channels=32)
#     # layer1: 32 * 1 = 32
#     # layer2: 64 * 1 = 64
#     # layer3: 128 * 1 = 128
#     # layer4: 256 * 1 = 256
#     # final_channels = 256
#     # 256 % nhead(4) == 0. 检查通过。

#     # (B, C, H, W) -> (B, 1, 512, 512)
#     inputs = torch.randn(1, 1, 512, 512).to(device) 
    
#     # 检查 forward 降维
#     # inputs_1d = torch.randn(1, 1, 512).to(device)
#     # flops_1d, params_1d = profile(model, (inputs_1d,), verbose=False)
#     # print(f'{name} (1D input) flops: {flops_1d}, params: {params_1d}')

#     flops, params = profile(model, (inputs,), verbose=False)
#     print(f'{name} (4D input) flops: {flops}, params: {params}')
    
#     out = model(inputs)
#     print(f'Output shape: {out.shape}')

#     # --- 顺便测试 Res_Inc_Net_two_layers (layers=(1,1,0,0), base_channels=32)
#     # layer1: 32 * 1 = 32
#     # layer2: 64 * 1 = 64
#     # layer3/4: 0
#     # final_channels = 64
#     # 64 % nhead(4) == 0. 检查通过。
#     print("\n--- Testing Res_Inc_Net_two_layers ---")
#     model_two = Res_Inc_Net_two_layers(num_classes=10, include_top=True).to(device)
#     flops, params = profile(model_two, (inputs,), verbose=False)
#     print(f'Res_Inc_Net_two_layers flops: {flops}, params: {params}')



if __name__ == "__main__":
    from torchsummary import summary
    from thop import profile
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for model_func, name in [
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