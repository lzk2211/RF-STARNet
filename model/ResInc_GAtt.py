import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

# ==========================================================================================
# Code from Res_Inc_Net.py
# ==========================================================================================

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


class ResNet1D(nn.Module):
    def __init__(self,
                 block,
                 blocks_num,
                 num_classes=1000,
                 include_top=True,
                 groups=1,
                 width_per_group=64):
        super(ResNet1D, self).__init__()
        self.include_top = include_top
        self.in_channel = 64
        self.groups = groups
        self.width_per_group = width_per_group

        # Input channel is 1
        self.conv1 = nn.Conv1d(1, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool1d(1)  # output size = 1
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

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
            x = self.fc(x)

        return x


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

def Res_Inc_Net_base(num_classes: int = 1000, include_top: bool = True):
    """ResNet-18 结构 (BasicBlock, [2,2,2,2])"""
    return Res_Inc_Net(BasicBlock1D, num_classes=num_classes, include_top=include_top,
        layers=(2, 2, 2, 2), base_channels=64, dropout=0.5)

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

# ==========================================================================================
# Code from RFProjectedAttentionClassifier.py
# ==========================================================================================

class RelativePositionBias(nn.Module):
    def __init__(self, seq_len: int, num_heads: int = 1):
        super().__init__()
        self.seq_len = int(seq_len)
        self.num_heads = int(num_heads)
        self.num_rel_pos = 2 * self.seq_len - 1
        self.relative_bias = nn.Parameter(torch.zeros(self.num_heads, self.num_rel_pos))
        nn.init.trunc_normal_(self.relative_bias, std=0.02)
        coords = torch.arange(self.seq_len)
        rel_pos = coords[None, :] - coords[:, None]
        rel_pos += self.seq_len - 1
        self.register_buffer("rel_pos_index", rel_pos)

    def forward(self, q_len: int, k_len: Optional[int] = None) -> torch.Tensor:
        if k_len is None:
            k_len = q_len
        if q_len == k_len == self.seq_len:
            return self.relative_bias[:, self.rel_pos_index]
        
        device = self.relative_bias.device
        S = self.seq_len
        denom_q = max(q_len - 1, 1)
        denom_k = max(k_len - 1, 1)
        q_pos_ref = torch.round(torch.linspace(0, S - 1, steps=q_len, device=device))
        k_pos_ref = torch.round(torch.linspace(0, S - 1, steps=k_len, device=device))
        rel = q_pos_ref[:, None] - k_pos_ref[None, :]
        rel = rel.clamp(min=-(S - 1), max=(S - 1)).to(torch.long)
        idx = (rel + (S - 1)).to(torch.long)
        return self.relative_bias[:, idx]

class GroupedProjectedAttentionRF(nn.Module):
    """
    改写的 GroupedProjectedAttention，专为输入 (B, F, T) 设计。
    内部视作 (B, T, D) with D=F（frequency bins as feature dim）并在 time 维度做压缩 k_len。
    支持当输入 T != seq_len 时对 Pk/Pv 做线性插值重采样。
    返回与输入相同的形状 (B, F, T)（即把通道放回 frequency-first 形式）。
    """

    def __init__(self,
                 freq_bins: int,
                 att_dim: int,
                 groups: int,
                 seq_len: int,
                 k_len: int,
                 use_pos: bool = True,
                 dropout: float = 0.0,
                 device: str = 'cuda'):
        """
        freq_bins: F (feature dim)
        att_dim: query/key dim per group (d_a)
        groups: number of groups (must divide freq_bins)
        seq_len: reference time length used to create Pk/Pv
        k_len: compressed key/value length
        """
        super().__init__()
        assert freq_bins % groups == 0, "freq_bins must be divisible by groups"
        self.freq_bins = freq_bins
        self.att_dim = att_dim
        self.groups = groups
        self.gdim = freq_bins // groups
        self.seq_len = seq_len
        self.k_len = k_len
        self.scale = 1.0 / math.sqrt(att_dim)
        self.use_pos = use_pos
        self.device = device

        # self.pos_encoding = PositionalEncoding(freq_bins, dropout)

        # per-group linear projections
        self.qs = nn.ModuleList([nn.Linear(self.gdim, att_dim, bias=False) for _ in range(groups)])
        self.ks = nn.ModuleList([nn.Linear(self.gdim, att_dim, bias=False) for _ in range(groups)])
        self.vs = nn.ModuleList([nn.Linear(self.gdim, self.gdim, bias=False) for _ in range(groups)])
        self.os = nn.ModuleList([nn.Linear(self.gdim, self.gdim, bias=False) for _ in range(groups)])

        # Pk, Pv: (seq_len, k_len) per group
        self.Pk = nn.ParameterList([nn.Parameter(torch.randn(seq_len, k_len) * (1.0 / math.sqrt(seq_len)))
                                    for _ in range(groups)])
        self.Pv = nn.ParameterList([nn.Parameter(torch.randn(seq_len, k_len) * (1.0 / math.sqrt(seq_len)))
                                    for _ in range(groups)])

        # 使用正弦-余弦位置编码（不再使用可学习 time_pos）
        # 保留 self.use_pos 开关，按需在 forward 中调用 position_coding(x)
        if use_pos:
            # (1, seq_len, 1) broadcastable for (B, T, D)
            self.time_pos = nn.Parameter(torch.zeros(1, seq_len, 1))
        else:
            self.register_parameter('time_pos', None)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.layer_norm = nn.LayerNorm(freq_bins)
        self.encoding_to_sa = nn.Sequential(nn.Linear(128, 128), nn.ReLU())

        self.rel_pos_bias = RelativePositionBias(seq_len=seq_len, num_heads=1)
    def _resample_P(self, P: torch.Tensor, target_T: int) -> torch.Tensor:
        """
        P: (seq_len, k_len)
        return: (target_T, k_len) by linear interpolation along seq_len dim if needed.
        """
        if target_T == self.seq_len:
            return P
        # P.T shape -> (k_len, seq_len) -> make batch (1, k_len, seq_len) and interpolate along last dim
        Pt = P.t().unsqueeze(0)  # (1, k_len, seq_len)
        # F.interpolate supports 3D input for linear mode: (N, C, L)
        Pt_resized = F.interpolate(Pt, size=target_T, mode='linear', align_corners=False)  # (1, k_len, target_T)
        Pt_resized = Pt_resized.squeeze(0)  # (k_len, target_T)
        return Pt_resized.t()  # -> (target_T, k_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, F, T)
        returns: out (B, F, T)
        """

        B, C, F, T = x.shape
        x = x.squeeze(1)  # (B, F, T)
        # x = x.permute(0, 2, 1)  # (B, T, F)
        assert F == self.freq_bins, f"Input freq dim {F} must equal freq_bins {self.freq_bins}"

        # if self.use_pos:
        #     if T == self.seq_len:
        #         pos = self.time_pos  # (1, seq_len, 1)
        #     else:
        #         # resample time_pos to new T
        #         pos = F.interpolate(self.time_pos.permute(0,2,1), size=T, mode='linear', align_corners=False)  # (1,1,T)
        #         pos = pos.permute(0,2,1)  # (1, T, 1)
        #     # x shape (B, F, T) -> transpose to (B, T, F) then add pos (broadcast on F)
        #     x_t = x.permute(0, 2, 1) + pos  # (B, T, F)
        # else:
        #     x_t = x.permute(0, 2, 1)  # (B, T, F)

        # x_t = self.pos_encoding(x.permute(0, 2, 1))  # (B, T, F)

        # 位置编码：使用给定的正弦-余弦编码函数，直接按当前 T/F 生成，无需插值
        # x_t = self.encoding_to_sa(x)
        # if self.use_pos:
        #     pos = position_coding(x_t).to(self.device)  # (1, T, F)
        #     x_t = x_t + pos

        x_t = x
        # 注意：当前张量形状为 (B, F, T)，LayerNorm(freq) 需作用在 F 维。
        # 由于 LayerNorm 期望最后一维为特征维，所以这里先转置到 (B, T, F)，归一化后再保持该布局用于注意力计算。
        x_t = x_t.permute(0, 2, 1)  # (B, T, F)
        x_t = self.layer_norm(x_t)  # normalize over F

        B, T2, D = x_t.shape  # 现在 T2 == 原始 T, D == F
        G = self.groups
        gdim = self.gdim

        # split into groups along feature dim: shape -> (B, G, T, gdim)
        xg = x_t.view(B, T2, G, gdim).permute(0, 2, 1, 3)  # (B, G, T, gdim)

        outs = []
        for g in range(G):
            xg_g = xg[:, g]  # (B, T, gdim)
            q = self.qs[g](xg_g)  # (B, T, att_dim)
            k = self.ks[g](xg_g)  # (B, T, att_dim)
            v = self.vs[g](xg_g)  # (B, T, gdim)

            # resample Pk/Pv if T2 != seq_len（与当前序列长度对齐）
            Pk_g = self._resample_P(self.Pk[g], T2)  # (T2, k_len)
            Pv_g = self._resample_P(self.Pv[g], T2)  # (T2, k_len)

            # compress along time: einsum -> k_red: (B, k_len, att_dim), v_red: (B, k_len, gdim)
            k_red = torch.einsum('bta,tk->bka', k, Pk_g)  # (B, k_len, att_dim)
            v_red = torch.einsum('btg,tk->bkg', v, Pv_g)  # (B, k_len, gdim)

            # attention: (B,T,att_dim) @ (B,att_dim,k_len) -> (B,T,k_len)
            att = torch.matmul(q, k_red.transpose(1, 2)) * self.scale

            # 相对位置偏置：生成 (1, T2, k_len)，与 att 形状对齐并按 (B, T2, k_len) 广播
            bias = self.rel_pos_bias(T2, self.k_len).to(att.device)  # (1, T2, k_len)
            att = att + bias.squeeze(0)  # (T2, k_len)

            att = torch.softmax(att, dim=-1)
            att = self.dropout(att)

            # out: (B, T, gdim)
            out_g = torch.matmul(att, v_red)
            out_g = self.os[g](out_g)  # (B, T, gdim)
            outs.append(out_g)

        # concat groups along feature axis -> (B, T, D)
        out = torch.cat(outs, dim=-1)  # (B, T, F)

        # transpose back to (B, F, T)
        out = out.permute(0, 2, 1).contiguous()  # (B, F, T)
        return out
    
# ==========================================================================================
# New Combined Model
# ==========================================================================================

class ResInc_GAtt(nn.Module):
    """
    A combined model that uses both ResNet1D and Projected Attention
    to extract features, which are then fused for final classification.
    """
    def __init__(self, num_classes=10, dropout_rate=0.0):
        super().__init__()
        
        # --- Branch 1: ResNet1D based feature extractor ---
        # We instantiate Res_Inc_Net without its final classification layer.
        self.resnet_features = Res_Inc_Net_base(num_classes=num_classes, include_top=False)
        
        # --- Branch 2: Projected Attention based feature extractor ---
        # We instantiate the core attention module.
        # Input dimensions (F=512, T=512) are set here.
        self.attention_features = GroupedProjectedAttentionRF(
            freq_bins=512,
            seq_len=128,  # Reference sequence length
            att_dim=32,   # Attention dimension per group
            groups=8,     # Number of attention groups
            k_len=64,     # Compressed key/value length
            dropout=0.1,
        )
        
        # --- Fusion and Classification Head ---
        # The ResNet branch outputs a 512-dim vector after pooling.
        # The Attention branch outputs a 512-dim vector after pooling.
        # Concatenated feature dimension is 512 + 512 = 1024.
        self.fusion_classifier = nn.Sequential(
            nn.Linear(512 + 512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the combined model.
        Args:
            x (torch.Tensor): Input tensor of shape (B, 1, 512, 512).
        Returns:
            torch.Tensor: Logits of shape (B, num_classes).
        """
        # Input validation
        if x.dim() != 4 or x.shape[1] != 1 or x.shape[2] != 512 or x.shape[3] != 512:
            raise ValueError(f"Expected input shape (B, 1, 512, 512), but got {x.shape}")

        # --- Branch 1: ResNet ---
        # Prepare input for ResNet1D: (B, 1, 512, 512) -> (B, 1, 512) by averaging the last dim.
        x_res = x.mean(dim=-1)
        
        # Extract features: output shape (B, 512, 16)
        res_out = self.resnet_features(x_res)
        
        # Global average pooling to get a fixed-size feature vector: (B, 512, 16) -> (B, 512)
        res_features = F.adaptive_avg_pool1d(res_out, 1).squeeze(-1)
        
        # --- Branch 2: Attention ---
        # The attention module expects (B, 1, 512, 512), so we can pass 'x' directly.
        
        # Extract features: output shape (B, 512, 512)
        att_out = self.attention_features(x)
        
        # Global average pooling over the time dimension: (B, 512, 512) -> (B, 512)
        att_features = att_out.mean(dim=-1)
        
        # --- Fusion and Classification ---
        # Concatenate features from both branches along the feature dimension.
        fused_features = torch.cat((res_features, att_features), dim=1) # Shape: (B, 1024)
        
        # Pass through the final classifier to get logits.
        logits = self.fusion_classifier(fused_features) # Shape: (B, num_classes)
        
        return logits


# ==========================================================================================
# Example Usage
# ==========================================================================================
if False:
    # # Define model parameters
    # BATCH_SIZE = 4
    # NUM_CLASSES = 10
    
    # # Check for GPU availability
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(f"Using device: {device}")
    
    # # Instantiate the model
    # model = ResInc_GAtt(num_classes=NUM_CLASSES)
    # model.to(device)
    
    # # Create a dummy input tensor with the specified dimensions
    # # Shape: (batch_size, channels, frequency_bins, time_steps)
    # dummy_input = torch.randn(BATCH_SIZE, 1, 512, 512).to(device)
    
    # print(f"Input shape: {dummy_input.shape}")
    
    # # Perform a forward pass
    # try:
    #     output_logits = model(dummy_input)
    #     print(f"Output shape: {output_logits.shape}")
        
    #     # Verify output shape
    #     assert output_logits.shape == (BATCH_SIZE, NUM_CLASSES)
    #     print("\nModel forward pass successful and output shape is correct.")
        
    # except Exception as e:
    #     print(f"\nAn error occurred during the forward pass: {e}")

    from torchsummary import summary
    from thop import profile
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = ResInc_GAtt(num_classes=10)
    model = model.to(device)

    # summary(model, input_size=(1, 512))
    inputs = torch.randn(1, 1, 512, 512).to(device)
    flops, params = profile(model, (inputs,))
    print('flops: ', flops, 'params: ', params)
