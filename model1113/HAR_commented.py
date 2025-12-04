"""
HAR_commented.py — HAR-Net 与 ResNet1D 变体的逐行中文超详细注释版本

设计目标与概览
- 提供 1D ResNet 风格主干网络，支持 BasicBlock、Bottleneck 与 HAR_Block 三种基本单元。
- HAR_Block（Harmonic-Aware Residual）通过“局部卷积 + 膨胀卷积（周期性感知）+ 通道注意力融合”提升对谐波/周期性结构的感知能力。
- 同时保留 include_top=True/False 以便作为分类器或特征提取 backbone 使用。
- 允许输入为 (B, C=1, F, T) 或 (B, C=1, L)；在 forward 中对 4D 输入内置 mean(dim=-1) -> (B,1,L) 的转换，兼容频时输入。

约定与数据形状
- 默认输入通道数 C=1（可根据任务改造）。
- 1D 卷积在“长度维度”上进行处理：输入张量形状统一为 (B, C, L)。
- ResNet stages（layer1~layer4）通道随层数递增，分辨率随 stride=2 下采样递减。

实现说明
- 本文件在原 HAR.py 基础上逐行添加详尽注释，核心逻辑保持一致，以便阅读与学习。
- 如无特别说明，groups=1（非分组卷积）；可在工厂函数中调整 base_channels / layers / dilations 来构造不同模型规模。
"""

from __future__ import annotations  # 允许注解中使用前置声明的类型（Python 3.7+ 安全）

import math  # 可能用于复杂调度或计算（当前实现中仅示例保留）
import torch
import torch.nn as nn


# ------------------------------------------------------------
# 基础模块 1: BasicBlock1D
# - 对应 ResNet-18/34 的基本残差单元（不带通道扩张），expansion=1。
# - 两个 3x1 的一维卷积（3 为 kernel_size；stride 由第一层控制），残差连接。
# - 支持 downsample 以匹配通道数或步幅差异。
# ------------------------------------------------------------
class BasicBlock1D(nn.Module):
    expansion = 1  # 残差块输出通道与“block内主分支最后一层 out_channel”的倍数关系

    def __init__(
        self,
        in_channel: int,  # 输入通道数
        out_channel: int,  # 主分支输出的基准通道数（最终输出通道=out_channel*expansion）
        stride: int = 1,   # 第一层卷积的步幅（决定是否下采样）
        downsample: nn.Module | None = None,  # 残差分支的下采样模块（通道/步幅对齐）
        groups: int = 1,  # 分组卷积的分组数（默认 1 表示普通卷积）
        width_per_group: int = 64,  # 预留参数（与 ResNeXt 等结构保持接口对齐）
        dilation: int = 1,  # 膨胀系数（当前 BasicBlock1D 未使用）
        **kwargs,
    ) -> None:
        super().__init__()

        # 第 1 个 3x1 Conv：可以带 stride 实现下采样；padding=1 保持长度不变（在 stride=1 情况下）
        self.conv1 = nn.Conv1d(
            in_channels=in_channel,
            out_channels=out_channel,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
            groups=groups,
        )
        self.bn1 = nn.BatchNorm1d(out_channel)  # BN 归一化
        self.relu = nn.ReLU()  # 非就地 ReLU，保持与部分层共享行为一致

        # 第 2 个 3x1 Conv：stride 固定为 1；保持通道不变
        self.conv2 = nn.Conv1d(
            in_channels=out_channel,
            out_channels=out_channel,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            groups=groups,
        )
        self.bn2 = nn.BatchNorm1d(out_channel)

        # 残差分支可选下采样模块，用于匹配主分支输出的形状（步幅与通道数）
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 保存恒等映射（残差分支的输入）
        identity = x
        # 如果提供了 downsample，则对 identity 做投影以匹配主分支输出的尺寸/通道
        if self.downsample is not None:
            identity = self.downsample(x)

        # 主分支：Conv -> BN -> ReLU
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # 主分支：Conv -> BN（此处先不 ReLU，等融合残差后再激活）
        out = self.conv2(out)
        out = self.bn2(out)

        # 残差相加
        out += identity
        # 融合后再 ReLU
        out = self.relu(out)

        return out


# ------------------------------------------------------------
# 基础模块 2: Bottleneck1D
# - 对应 ResNet-50/101/152 的“瓶颈”残差单元，expansion=4。
# - 结构：1x1 降维 -> 3x1 卷积 -> 1x1 升维；可减少 FLOPs。
# - 也支持 downsample 以匹配残差。
# ------------------------------------------------------------
class Bottleneck1D(nn.Module):
    expansion = 4  # 最终输出通道 = out_channel * 4

    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
        groups: int = 1,
        width_per_group: int = 64,
        dilation: int = 1,
        **kwargs,
    ) -> None:
        super().__init__()

        # width 是瓶颈中的中间通道数（可与 ResNeXt 接口兼容，保留 groups/width_per_group）
        width = int(out_channel * (width_per_group / 64.0)) * groups

        # 1x1 降维（或特征重排），不改变时间长度
        self.conv1 = nn.Conv1d(
            in_channels=in_channel,
            out_channels=width,
            kernel_size=1,
            stride=1,
            bias=False,
            groups=groups,
        )
        self.bn1 = nn.BatchNorm1d(width)
        self.relu = nn.ReLU(inplace=True)  # 瓶颈块常用 inplace=True 以节省内存

        # 3x1 卷积：可设置 stride 实现下采样
        self.conv2 = nn.Conv1d(
            in_channels=width,
            out_channels=width,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
            groups=groups,
        )
        self.bn2 = nn.BatchNorm1d(width)

        # 1x1 升维：输出通道恢复到 out_channel * expansion
        self.conv3 = nn.Conv1d(
            in_channels=width,
            out_channels=out_channel * self.expansion,
            kernel_size=1,
            stride=1,
            bias=False,
            groups=groups,
        )
        self.bn3 = nn.BatchNorm1d(out_channel * self.expansion)

        # 残差分支下采样（如果需要）
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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


# ------------------------------------------------------------
# 核心模块 3: HAR_Block
# - 关键思想：
#   1) 局部分支：普通 3x1 卷积，感受野较小，聚焦局部纹理/边缘。
#   2) 周期分支：带 dilation 的 3x1 卷积，感受野更大，用于感知更大尺度的周期规律。
#   3) 通道级软注意力：根据融合特征的全局统计，自适应给两条分支分配权重（softmax 保证和为 1）。
#   4) 残差连接：融合后仍与输入相加，保留梯度通路与恒等信息。
# ------------------------------------------------------------
class HAR_Block(nn.Module):
    expansion = 1  # 与 BasicBlock 一致，保持通道不扩张

    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
        groups: int = 1,
        width_per_group: int = 64,  # 为接口一致性保留
        dilation: int = 4,          # 周期分支的膨胀系数，默认较大以扩大感受野
        reduction: int = 16,        # 注意力的压缩比（SE 风格）
    ) -> None:
        super().__init__()

        self.relu = nn.ReLU(inplace=True)  # 共享 ReLU
        self.downsample = downsample       # 残差分支的下采样（如需要）

        # 分支 A：局部 3x1 卷积（dilation=1），可带 stride 下采样
        self.conv_local = nn.Conv1d(
            in_channels=in_channel,
            out_channels=out_channel,
            kernel_size=3,
            stride=stride,
            padding=1,          # 普通 3x1 保持长度
            dilation=1,
            groups=groups,
            bias=False,
        )
        self.bn_local = nn.BatchNorm1d(out_channel)

        # 分支 B：周期/谐波感知分支 —— 3x1 膨胀卷积
        # padding 选择：简单地使用 same-padding 近似 ((k-1)*dilation)//2
        periodic_padding = (3 - 1) * dilation // 2
        self.conv_periodic = nn.Conv1d(
            in_channels=in_channel,
            out_channels=out_channel,
            kernel_size=3,
            stride=stride,
            padding=periodic_padding,
            dilation=dilation,  # 关键：扩大感受野，捕获更长周期模式
            groups=groups,
            bias=False,
        )
        self.bn_periodic = nn.BatchNorm1d(out_channel)

        # 融合后的注意力：使用 GAP -> 两层全连接 -> 2 类 softmax，输出 [w_local, w_periodic]
        self.gap = nn.AdaptiveAvgPool1d(1)  # 对时间维做自适应平均（输出形状 B,C,1）
        squeeze_channels = max(1, out_channel // reduction)  # 压缩后的通道数至少为 1
        self.fc_squeeze = nn.Linear(out_channel, squeeze_channels)  # C -> C/r
        self.fc_excite = nn.Linear(squeeze_channels, 2)             # C/r -> 2（两条分支权重）
        self.softmax = nn.Softmax(dim=1)                            # 在“分支维”做归一化

        self.bn_out = nn.BatchNorm1d(out_channel)  # 融合后 BN

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)  # 残差分支对齐

        # 局部分支：Conv -> BN
        x_local = self.conv_local(x)
        x_local = self.bn_local(x_local)

        # 周期分支：带 dilation 的 Conv -> BN
        x_periodic = self.conv_periodic(x)
        x_periodic = self.bn_periodic(x_periodic)

        # 两分支先相加再激活，得到融合特征（也可考虑 concat+1x1，但此处用更轻量的加法）
        x_fused = self.relu(x_local + x_periodic)

        # 全局上下文：GAP 之后为 (B, C, 1)，去掉最后一维 -> (B, C)
        u = self.gap(x_fused).squeeze(-1)
        # SE 风格两层 MLP：C -> C/r -> 2
        z = self.fc_squeeze(u)
        z = self.relu(z)
        weights = self.softmax(self.fc_excite(z))  # 在维度=1 做 softmax，得到 (B,2)

        # 将两路权重形状从 (B,2) 变成 (B,1,1)，以便对 (B,C,L) 做逐通道广播
        w_local = weights[:, 0].unsqueeze(-1).unsqueeze(-1)
        w_periodic = weights[:, 1].unsqueeze(-1).unsqueeze(-1)

        # 加权融合两分支输出
        out = (x_local * w_local) + (x_periodic * w_periodic)
        out = self.bn_out(out)

        # 残差相加并激活
        out += identity
        out = self.relu(out)

        return out


# ------------------------------------------------------------
# 主干网络 4: ResNet1D
# - 将上面的 block 级模块（Basic/Bottleneck/HAR_Block）堆叠为四个 stage。
# - 支持 dilations_per_layer：为每个 stage 指定每个 block 的膨胀系数列表（HAR_Block 有效）。
# - include_top 控制是否包含分类头（avgpool + fc）。
# ------------------------------------------------------------
class ResNet1D(nn.Module):
    def __init__(
        self,
        block: type[nn.Module],             # 残差块类（BasicBlock1D / Bottleneck1D / HAR_Block）
        blocks_num: list[int],               # 每个 stage 的 block 个数，例如 [2,2,2,2]
        num_classes: int = 1000,             # 分类数（include_top=True 时生效）
        include_top: bool = True,            # 是否包含分类头
        groups: int = 1,                     # 分组卷积组数（默认 1）
        width_per_group: int = 64,           # 与 ResNeXt 接口保持一致
        base_channels: int = 64,             # layer1 的通道基数
        width_mult: float = 1.0,             # 宽度放缩系数（便于快速缩放模型大小）
        stem_channels: int | None = None,    # stem 输出通道（默认等于 base_channels）
        dropout: float = 0.0,                # 分类头前的 dropout 概率
        dilations_per_layer: dict | None = None,  # 每层的 dilation 列表，例如 {"layer1":[2,4], ...}
    ) -> None:
        super().__init__()

        # 头部/通道配置
        self.include_top = include_top
        self.in_channel = stem_channels if stem_channels is not None else base_channels
        self.groups = groups
        self.width_per_group = width_per_group
        self.base_channels = base_channels
        self.width_mult = float(width_mult)
        self.dropout_p = float(dropout)

        # Stem：7x1 Conv -> BN -> ReLU -> 3x1 MaxPool（stride=2）
        self.conv1 = nn.Conv1d(1, self.in_channel, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # 内部工具：向上取整到 groups 的整数倍（与部分分组卷积设置对齐，当前 groups=1 时等价不变）
        def _make_divisible(v: int, divisor: int) -> int:
            if divisor <= 1:
                return int(v)
            v = int(v)
            return max(divisor, (v + divisor - 1) // divisor * divisor)

        # 四个 stage 的基准倍数（类似于 64,128,256,512，但这里以 base_channels 为起点）
        stage_base = [1, 2, 4, 8]
        stage_channels = [
            _make_divisible(int(self.base_channels * b * self.width_mult), self.groups)
            for b in stage_base
        ]

        # 膨胀系数配置，若未指定则默认都为 1
        if dilations_per_layer is None:
            dilations_per_layer = {}
        default_dilation = 1

        def get_dilations(stage_name: str, num: int) -> list[int]:
            return dilations_per_layer.get(stage_name, [default_dilation] * num)

        # 为四个 stage 准备各自的 dilation 列表
        dilations_l1 = get_dilations("layer1", blocks_num[0])
        dilations_l2 = get_dilations("layer2", blocks_num[1])
        dilations_l3 = get_dilations("layer3", blocks_num[2])
        dilations_l4 = get_dilations("layer4", blocks_num[3])

        # 构建四个 stage；其中 layer2~4 默认 stride=2 进行下采样
        self.layer1 = self._make_layer(block, stage_channels[0], blocks_num[0], dilations=dilations_l1)
        self.layer2 = self._make_layer(block, stage_channels[1], blocks_num[1], stride=2, dilations=dilations_l2)
        self.layer3 = self._make_layer(block, stage_channels[2], blocks_num[2], stride=2, dilations=dilations_l3)
        self.layer4 = self._make_layer(block, stage_channels[3], blocks_num[3], stride=2, dilations=dilations_l4)

        # 分类头：include_top=True 时启用；否则使用 Identity 以便当作 backbone输出特征
        self.dropout = nn.Dropout(p=self.dropout_p) if self.dropout_p and include_top else nn.Identity()
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool1d(1)  # 时间维自适应平均 -> (B,C,1)
            final_channels = stage_channels[3] * block.expansion
            self.fc = nn.Linear(final_channels, num_classes)

        # 卷积权重初始化：Kaiming Normal，适配 ReLU 非线性
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

        # 可学习阈值（分类后处理用，当前示例未直接使用，作为接口预留）
        self.thresholds = nn.Parameter(torch.ones(num_classes) * 0.5)

    def _make_layer(
        self,
        block: type[nn.Module],
        channel: int,           # 当前 stage 的基准通道数（未乘 expansion 前）
        block_num: int,         # 当前 stage 的 block 个数
        stride: int = 1,        # 当前 stage 的第一层 stride（决定是否下采样）
        dilations: list[int] | None = None,  # 每个 block 的膨胀系数列表
    ) -> nn.Sequential:
        # 处理 dilation 列表：默认全 1；长度需匹配 block_num
        if dilations is None:
            dilations = [1] * block_num
        if len(dilations) != block_num:
            raise ValueError(f"dilations 列表长度 {len(dilations)} 必须等于 block_num {block_num}")

        # 如果 stride!=1 或者 通道数不匹配，则需要对残差分支做下采样投影
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(
                    self.in_channel,
                    channel * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                    groups=self.groups,
                ),
                nn.BatchNorm1d(channel * block.expansion),
            )

        layers: list[nn.Module] = []
        # 第一个 block：可能携带 stride/downsample
        layers.append(
            block(
                self.in_channel,
                channel,
                downsample=downsample,
                stride=stride,
                groups=self.groups,
                width_per_group=self.width_per_group,
                dilation=dilations[0],
            )
        )

        # 更新当前网络的“输入通道”记录为本 stage 的输出通道（乘以 expansion）
        self.in_channel = channel * block.expansion

        # 余下 block：stride=1，通道保持一致
        for i in range(1, block_num):
            layers.append(
                block(
                    self.in_channel,
                    channel,
                    groups=self.groups,
                    width_per_group=self.width_per_group,
                    dilation=dilations[i],
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 兼容输入为 (B,1,F,T) 的张量：对最后一维做均值，得到 (B,1,F)；
        # 在 1D 处理中，F 作为“长度 L”进入后续网络。
        if x.dim() == 4:
            x = x.mean(dim=-1)  # (B, 1, F, T) -> (B, 1, F)

        # Stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # 四个 stage 依次堆叠
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # 分类头（可选）
        if self.include_top:
            x = self.avgpool(x)      # (B, C, L) -> (B, C, 1)
            x = torch.flatten(x, 1)  # -> (B, C)
            x = self.dropout(x)
            x = self.fc(x)           # -> (B, num_classes)

        return x


# ------------------------------------------------------------
# 工厂函数：便捷创建不同规模/结构的网络
# - Res_Inc_Net：通用工厂，接收 block 类型与层配置等参数。
# - 提供若干具体实例：Tiny/Small/Base/Large（BasicBlock）、Res_Inc_Net_50（Bottleneck）。
# - HAR_Net_*：使用 HAR_Block 的变体，内置一份 dilation 调度示例。
# ------------------------------------------------------------
def Res_Inc_Net(
    block: type[nn.Module],
    num_classes: int = 1000,
    include_top: bool = True,
    layers: tuple[int, int, int, int] = (1, 1, 1, 1),
    groups: int = 32,
    base_channels: int = 64,
    width_mult: float = 1.0,
    stem_channels: int | None = None,
    dropout: float = 0.5,
    dilations_per_layer: dict | None = None,
) -> ResNet1D:
    return ResNet1D(
        block,
        list(layers),
        num_classes=num_classes,
        include_top=include_top,
        groups=groups,
        width_per_group=4,  # 与原实现一致，作为内部“width-per-group”基准
        base_channels=base_channels,
        width_mult=width_mult,
        stem_channels=stem_channels,
        dropout=dropout,
        dilations_per_layer=dilations_per_layer,
    )


# -------- BasicBlock 实例族（近似 ResNet18/34 级别） --------
def Res_Inc_Net_tiny(num_classes: int = 1000, include_top: bool = True) -> ResNet1D:
    return Res_Inc_Net(
        BasicBlock1D,
        num_classes=num_classes,
        include_top=include_top,
        layers=(1, 1, 1, 1),
        base_channels=32,
        dropout=0.3,
    )


def Res_Inc_Net_small(num_classes: int = 1000, include_top: bool = True) -> ResNet1D:
    return Res_Inc_Net(
        BasicBlock1D,
        num_classes=num_classes,
        include_top=include_top,
        layers=(1, 1, 1, 1),
        base_channels=64,
        dropout=0.5,
    )


def Res_Inc_Net_base(
    num_classes: int = 1000,
    include_top: bool = True,
    dropout: float = 0.5,
) -> ResNet1D:
    """近似 ResNet-18 结构：BasicBlock，[2,2,2,2]"""
    return Res_Inc_Net(
        BasicBlock1D,
        num_classes=num_classes,
        include_top=include_top,
        layers=(2, 2, 2, 2),
        base_channels=64,
        dropout=dropout,
    )


def Res_Inc_Net_large(num_classes: int = 1000, include_top: bool = True) -> ResNet1D:
    """近似 ResNet-34 结构：BasicBlock，[3,4,6,3]"""
    return Res_Inc_Net(
        BasicBlock1D,
        num_classes=num_classes,
        include_top=include_top,
        layers=(3, 4, 6, 3),
        base_channels=64,
        dropout=0.5,
    )


# -------- Bottleneck 实例（近似 ResNet-50） --------
def Res_Inc_Net_50(num_classes: int = 1000, include_top: bool = True) -> ResNet1D:
    """近似 ResNet-50 结构：Bottleneck，[3,4,6,3]"""
    return Res_Inc_Net(
        Bottleneck1D,
        num_classes=num_classes,
        include_top=include_top,
        layers=(3, 4, 6, 3),
        base_channels=64,
        dropout=0.5,
    )


# -------- HAR-Net 实例族（使用 HAR_Block） --------
def HAR_Net_tiny(num_classes: int = 1000, include_top: bool = True) -> ResNet1D:
    """HAR-Net-Tiny：HAR_Block，[1,1,1,1]，base_channels=32"""
    layers_config = (1, 1, 1, 1)
    dilation_schedule = {  # 为每个 stage 的 block 指定 dilation
        "layer1": [2],
        "layer2": [4],
        "layer3": [8],
        "layer4": [16],
    }
    return Res_Inc_Net(
        HAR_Block,
        num_classes=num_classes,
        include_top=include_top,
        layers=layers_config,
        base_channels=32,
        dropout=0.3,
        dilations_per_layer=dilation_schedule,
    )


def HAR_Net_small(num_classes: int = 1000, include_top: bool = True) -> ResNet1D:
    """HAR-Net-Small：HAR_Block，[1,1,1,1]，base_channels=64"""
    layers_config = (1, 1, 1, 1)
    dilation_schedule = {
        "layer1": [2],
        "layer2": [4],
        "layer3": [8],
        "layer4": [16],
    }
    return Res_Inc_Net(
        HAR_Block,
        num_classes=num_classes,
        include_top=include_top,
        layers=layers_config,
        base_channels=64,
        dropout=0.5,
        dilations_per_layer=dilation_schedule,
    )


def HAR_Net_base(
    num_classes: int = 1000,
    include_top: bool = True,
    dropout: float = 0.5,
) -> ResNet1D:
    """HAR-Net-Base：HAR_Block，[2,2,2,2]（相当于 ResNet-18 深度）"""
    layers_config = (2, 2, 2, 2)
    # 这里示例展示一种在不同 stage 内部变换 dilation 的策略；
    # 可根据任务经验自行调整（例如保持单调递增/固定/循环等）。
    dilation_schedule = {
        "layer1": [2, 4],
        "layer2": [8, 16],
        "layer3": [32, 2],
        "layer4": [4, 8],
    }
    return Res_Inc_Net(
        HAR_Block,
        num_classes=num_classes,
        include_top=include_top,
        layers=layers_config,
        base_channels=64,
        dropout=dropout,
        dilations_per_layer=dilation_schedule,
    )


def HAR_Net_large(num_classes: int = 1000, include_top: bool = True) -> ResNet1D:
    """HAR-Net-Large：HAR_Block，[3,4,6,3]（相当于 ResNet-34 深度）"""
    layers_config = (3, 4, 6, 3)
    dilation_schedule = {
        "layer1": [2, 4, 8],
        "layer2": [2, 4, 8, 16],
        "layer3": [2, 4, 8, 16, 32, 8],
        "layer4": [2, 4, 16],
    }
    return Res_Inc_Net(
        HAR_Block,
        num_classes=num_classes,
        include_top=include_top,
        layers=layers_config,
        base_channels=64,
        dropout=0.5,
        dilations_per_layer=dilation_schedule,
    )


# ------------------------------------------------------------
# 自测（仅在直接运行本文件时执行，import 不会触发）
# - 轻量验证：在 CPU 上做一次前向，确认形状与计算图无误。
# - 如需 FLOPs/params，建议在上层脚本中按需使用 thop.profile（避免此处强依赖）。
# ------------------------------------------------------------
if __name__ == "__main__":
    # 选择一个小模型做 smoke test
    model = HAR_Net_small(num_classes=10, include_top=True)
    model.eval()

    # 构造 (B, C=1, F, T) 输入，并让 forward 自动均值到 (B, 1, F)
    x = torch.randn(1, 1, 512, 512)
    with torch.no_grad():
        y = model(x)
    print("HAR_Net_small output:", y.shape)
