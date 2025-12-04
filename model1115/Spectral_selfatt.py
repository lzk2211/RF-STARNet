import torch.nn as nn
import torch
import torch.nn.functional as F  # <-- 导入 F，用于 softmax
from torchsummary import summary
from thop import profile
import math

# ======================================================================
# 模块 0: SELayer1D (来自您的原始文件)
# ======================================================================
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

# ======================================================================
# 模块 1: BasicBlock1D (来自您的原始文件, 保留对比)
# ======================================================================
class BasicBlock1D(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, groups=1, 
                 width_per_group=64,
                 dilation_levels=None
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

# ======================================================================
# 模块 2: MultiDilationBasicBlock1D (来自您的原始文件)
# ======================================================================
class MultiDilationBasicBlock1D(nn.Module):
    """
    创新的多尺度空洞卷积块 (Multi-Dilation Basic Block)
    """
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, groups=1, 
                 width_per_group=64, 
                 dilation_levels=[1, 2, 4, 8]
                 ):
        super(MultiDilationBasicBlock1D, self).__init__()
        
        self.num_dilation_levels = len(dilation_levels)
        
        if in_channel % self.num_dilation_levels != 0 or out_channel % self.num_dilation_levels != 0:
            raise ValueError(f"in_channel ({in_channel}) 和 out_channel ({out_channel}) "
                             f"必须能被 num_dilation_levels ({self.num_dilation_levels}) 整除。")
        if groups % self.num_dilation_levels != 0:
            raise ValueError(f"groups ({groups}) 必须能被 num_dilation_levels ({self.num_dilation_levels}) 整除。")
            
        self.in_ch_path = in_channel // self.num_dilation_levels
        self.out_ch_path = out_channel // self.num_dilation_levels
        self.groups_per_path = groups // self.num_dilation_levels

        self.conv1_paths = nn.ModuleList()
        for dilation in dilation_levels:
            padding = dilation 
            self.conv1_paths.append(nn.Conv1d(in_channels=self.in_ch_path, 
                                              out_channels=self.out_ch_path,
                                              kernel_size=3, 
                                              stride=stride, 
                                              padding=padding, 
                                              bias=False, 
                                              groups=self.groups_per_path, 
                                              dilation=dilation))

        self.bn1 = nn.BatchNorm1d(out_channel)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv1d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, 
                               stride=1, 
                               padding=1, 
                               bias=False, 
                               groups=groups)
        self.bn2 = nn.BatchNorm1d(out_channel)
        self.downsample = downsample

        self.SELayer1D = SELayer1D(out_channel, reduction=16)

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        x_split = torch.split(x, self.in_ch_path, dim=1)
        path_outs = []
        for i in range(self.num_dilation_levels):
            path_outs.append(self.conv1_paths[i](x_split[i]))
        
        out = torch.cat(path_outs, dim=1)

        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.SELayer1D(out)
        out += identity
        out = self.relu(out)

        return out

# ======================================================================
# 模块 3: Bottleneck1D (来自您的原始文件, 保留)
# ======================================================================
class Bottleneck1D(nn.Module):
    expansion = 4 

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, groups=1, width_per_group=64,
                 dilation_levels=None
                 ):
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

# ======================================================================
# 模块 4: ResNet1D (来自您的原始文件, 通用)
# ======================================================================
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
                 dilation_levels: list = None
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

        self.conv1 = nn.Conv1d(1, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        def _make_divisible(v: int, divisor: int) -> int:
            if divisor <= 1:
                return int(v)
            v = int(v)
            divisor = max(divisor, len(self.dilation_levels)) 
            return max(divisor, (v + divisor - 1) // divisor * divisor)

        stage_base = [1, 2, 4, 8]
        stage_channels = [
            _make_divisible(int(self.base_channels * b * self.width_mult), self.groups)
            for b in stage_base
        ]
        
        stage_channels = [
             _make_divisible(c, len(self.dilation_levels)) for c in stage_channels
        ]
        
        
        self.layer1 = self._make_layer(block, stage_channels[0], self.blocks_num[0])
        self.layer2 = self._make_layer(block, stage_channels[1], self.blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, stage_channels[2], self.blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, stage_channels[3], self.blocks_num[3], stride=2)

        self.dropout_layer = nn.Dropout(p=self.dropout_p) if self.dropout_p and include_top else nn.Identity()
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool1d(1)
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
        if block_num <= 0:
            return nn.Identity()

        downsample = None
        
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
        # x 初始 shape: (B, 1, 512, 512)
        x = x.mean(dim=-1) # (B, 1, 512)

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
            x = self.dropout_layer(x)
            x = self.fc(x)

        # 如果 include_top=False, 将直接返回 layer4 (或最后一个活动层) 的输出
        # 对于 Res_Inc_Net_mini, layer3/4 是 nn.Identity
        # 所以返回的是 layer2 的输出, shape [B, 128, 64]
        return x

# ======================================================================
# 模块 5: 工厂函数 (来自您的原始文件, 保留)
# ======================================================================
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

def Res_Inc_Net_mini(num_classes: int = 1000, include_top: bool = True):
    return Res_Inc_Net(MultiDilationBasicBlock1D, num_classes=num_classes, include_top=include_top,
        layers=(1, 1, 0, 0), base_channels=64, groups=32, dropout=0.3)

def Res_Inc_Net_tiny(num_classes: int = 1000, include_top: bool = True):
    return Res_Inc_Net(MultiDilationBasicBlock1D, num_classes=num_classes, include_top=include_top,
        layers=(1, 1, 1, 1), base_channels=32, groups=32, dropout=0.3)

def Res_Inc_Net_small(num_classes: int = 1000, include_top: bool = True):
    return Res_Inc_Net(MultiDilationBasicBlock1D, num_classes=num_classes, include_top=include_top,
        layers=(1, 1, 1, 1), base_channels=64, groups=32, dropout=0.5)

def Res_Inc_Net_base(num_classes: int = 1000, include_top: bool = True, dropout: float = 0.5):
    return Res_Inc_Net(MultiDilationBasicBlock1D, num_classes=num_classes, include_top=include_top,
        layers=(2, 2, 2, 2), base_channels=64, groups=32, dropout=dropout)

def Res_Inc_Net_large(num_classes: int = 1000, include_top: bool = True):
    return Res_Inc_Net(MultiDilationBasicBlock1D, num_classes=num_classes, include_top=include_top,
        layers=(3, 4, 6, 3), base_channels=64, groups=32, dropout=0.5)

def Res_Inc_Net_50(num_classes: int = 1000, include_top: bool = True):
    return Res_Inc_Net(Bottleneck1D, num_classes=num_classes, include_top=include_top,
        layers=(3, 4, 6, 3), base_channels=64, groups=32, dropout=0.5)

def Res_Inc_Net_101(num_classes: int = 1000, include_top: bool = True):
    return Res_Inc_Net(Bottleneck1D, num_classes=num_classes, include_top=include_top,
        layers=(3, 4, 23, 3), base_channels=64, groups=32, dropout=0.5)

def Res_Inc_Net_two_layers(num_classes: int = 1000, include_top: bool = True):
    return Res_Inc_Net(
        MultiDilationBasicBlock1D,
        num_classes=num_classes,
        include_top=include_top,
        layers=(1, 1, 0, 0),
        base_channels=32,
        groups=32,
        dropout=0.3,
    )

# ======================================================================
# 模块 6: 【新增】轻量化自注意力模块 (从零实现)
# ======================================================================
class LightSelfAttention1D(nn.Module):
    def __init__(self, embed_dim):
        """
        轻量级单头自注意力 (不使用 torch.nn.MultiheadAttention)
        :param embed_dim: 输入特征维度 (C)
        """
        super().__init__()
        self.embed_dim = embed_dim
        # 缩放因子
        self.scale = embed_dim ** -0.5 

        # 1. 使用一个线性层统一生成 Q, K, V
        self.to_qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        
        # 2. 输出线性层
        self.to_out = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        # x 输入 shape: (B, L, C) - [Batch, Length, Channels]
        B, L, C = x.shape
        
        # 1. 计算 Q, K, V
        # (B, L, C) -> (B, L, C*3)
        qkv = self.to_qkv(x)
        
        # 2. 拆分 Q, K, V
        # 均得到 (B, L, C)
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        # 3. 计算注意力得分 (Q @ K_T)
        # (B, L, C) @ (B, C, L) -> (B, L, L)
        attn_scores = torch.bmm(q, k.transpose(1, 2)) * self.scale
        
        # 4. Softmax
        attn_probs = F.softmax(attn_scores, dim=-1)

        # 5. 计算加权和 (Scores @ V)
        # (B, L, L) @ (B, L, C) -> (B, L, C)
        context = torch.bmm(attn_probs, v)

        # 6. 输出投影
        # (B, L, C) -> (B, L, C)
        out = self.to_out(context)
        
        # 7. 添加残差连接 (标准 Transformer 结构)
        return x + out

# ======================================================================
# 模块 7: 【新增】结合 ResNet 特征与注意力的新模型
# ======================================================================
class SpectralAttentionNet(nn.Module):
    def __init__(self, num_classes=10, dropout=0.3):
        super().__init__()
        
        # 1. 骨干网络 (Backbone)
        # 使用 Res_Inc_Net_mini 工厂函数, 但设置 include_top=False
        # 这将返回一个 ResNet1D 实例, 其 forward() 会返回最后一个 stage 的特征
        # layers=(1, 1, 0, 0), 最后一个 stage 是 layer2
        self.backbone = Res_Inc_Net_mini(num_classes=num_classes, include_top=False)
        
        # 分析 Res_Inc_Net_mini:
        # base_channels=64, 对应 stage_channels[0]=64
        # stage_channels[1]=128
        # layer1 (s=1) 输出: [B, 64, 128]
        # layer2 (s=2) 输出: [B, 128, 64]
        # 所以, backbone 的输出特征维度 (C) 是 128
        feature_dim = 128
        
        # 2. 轻量化自注意力模块
        self.attention = LightSelfAttention1D(embed_dim=feature_dim)
        
        # 3. 轻量化分类头
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2), # 128 -> 64
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(feature_dim // 2, num_classes)  # 64 -> num_classes
        )

    def forward(self, x):
        # x 初始 shape: (B, 1, 512, 512)
        
        # 1. 通过骨干网络提取特征
        # backbone.forward() 已经包含了 x.mean(dim=-1) 操作
        # include_top=False, 它将返回 layer2 的输出
        # x shape: (B, 128, 64)  [B, C, L]
        x = self.backbone(x)
        
        # 2. 调整 shape 以适应注意力模块
        # (B, C, L) -> (B, L, C)
        # (B, 128, 64) -> (B, 64, 128)
        x = x.transpose(1, 2)
        
        # 3. 通过自注意力模块
        # (B, 64, 128) -> (B, 64, 128)
        x = self.attention(x)
        
        # 4. 聚合特征 (在序列长度 L 维度上做全局平均池化)
        # (B, 64, 128) -> (B, 128)
        x = x.mean(dim=1)
        
        # 5. 通过分类头
        # (B, 128) -> (B, num_classes)
        logits = self.classifier(x)
        
        return logits

# ======================================================================
# 模块 8: 测试 (更新为测试新模型)
# ======================================================================
if __name__ == "__main__":
    from torchsummary import summary
    from thop import profile
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"--- Testing SpectralAttentionNet (基于 Res_Inc_Net_mini 前两层 + 自注意力) ---")
    
    # 实例化我们的新模型
    model = SpectralAttentionNet(num_classes=10)
    model = model.to(device)
    
    # 输入 shape 仍然是 (B, 1, 512, 512)
    inputs = torch.randn(1, 1, 512, 512).to(device)
    
    flops, params = profile(model, (inputs,), verbose=False)
    print(f'SpectralAttentionNet flops: {flops}, params: {params}')
    
    # --- (可选) 打印原始 Res_Inc_Net_mini 作为对比 ---
    print(f"\n--- Testing 原始 Res_Inc_Net_mini (用于对比) ---")
    model_mini = Res_Inc_Net_mini(num_classes=10, include_top=True)
    model_mini = model_mini.to(device)
    flops_mini, params_mini = profile(model_mini, (inputs,), verbose=False)
    print(f'Res_Inc_Net_mini flops: {flops_mini}, params: {params_mini}')
