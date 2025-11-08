# S3R_with_ResInc.py
# ------------------
#
# 这是一个修改后的版本，它将 S3R.py 中的 2D CNN 特征提取器
# 替换为了 Res_Inc_Net.py 中的 1D ResNet 提取器。
#
# 主要改动：
# 1. 引入了 BasicBlock1D, ResNet1D, 和 Res_Inc_Net 的定义。
# 2. 在 NET.__init__ 中：
#    - 删除了 self.encoder_d1, self.encoder_d3, self.encoder_d5。
#    - 删除了 self.encoder_to_semantic。
#    - 添加了 self.res_inc_extractor，它是一个 Res_Inc_Net 实例，
#      其输出维度被设置为 semantic_dim。
# 3. 在 NET.forward 中：
#    - 保留了 y 和 z 分支（Transformer）的完整逻辑。
#    - 替换了 x 分支（2D CNN）的逻辑。
#    - 新逻辑通过 adaptive_avg_pool2d 将 2D 输入 [B, 1, 512, 512]
#      转换为 1D 输入 [B, 1, 512]。
#    - 将此 1D 张量送入 self.res_inc_extractor 以获取 x_features。
#    - 最后将 x_features, y, z 三个语义向量拼接。
#
import torch
from torch import nn
import torch.nn.functional as F

# ======================================================
#  代码块 1: 从 Res_Inc_Net.py 导入
# ======================================================

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



# ======================================================
#  代码块 2: 从 S3R.py 导入并修改
# ======================================================

def position_coding(x):
    num_token, num_dims = x.size(-2), x.size(-1)
    p = torch.zeros((1, num_token, num_dims))
    t = torch.arange(num_token, dtype=torch.float32).reshape(-1, 1) / \
        torch.pow(1e4, torch.arange(0, num_dims, 2, dtype=torch.float32) / num_dims)
    p[:, :, 0::2] = torch.sin(t)
    p[:, :, 1::2] = torch.cos(t)
    return p


class NET(nn.Module):
    def __init__(self, in_channels, input_size, semantic_dim, num_class, device):
        super(NET, self).__init__()
        self.input_size = input_size                     # [T, W]
        self.semantic_dim = semantic_dim
        self.num_class = num_class
        self.device = device
        self.in_channels = in_channels

        # --- Transformer (y) 分支 (保留) ---
        self.SA_1 = nn.TransformerEncoderLayer(d_model=64, nhead=8, batch_first=True, dim_feedforward=256)
        self.SA_1 = nn.TransformerEncoder(self.SA_1, num_layers=3)
        self.encoding_to_sa1 = nn.Sequential(
            nn.Linear(self.input_size[1], 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.sa1_to_semantic = nn.Sequential(
            nn.Linear(int(self.input_size[0] * 64), 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, self.semantic_dim, bias=False),
            nn.BatchNorm1d(self.semantic_dim),
            nn.ReLU()
        )

        # --- Transformer (z) 分支 (保留) ---
        self.SA_2 = nn.TransformerEncoderLayer(d_model=64, nhead=8, batch_first=True, dim_feedforward=256)
        self.SA_2 = nn.TransformerEncoder(self.SA_2, num_layers=3)
        self.encoding_to_sa2 = nn.Sequential(
            nn.Linear(self.input_size[0], 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.sa2_to_semantic = nn.Sequential(
            nn.Linear(int(self.input_size[1] * 64), 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, self.semantic_dim, bias=False),
            nn.BatchNorm1d(self.semantic_dim),
            nn.ReLU()
        )

        # --- (被替换) 原始的 2D CNN (x) 分支 ---
        # self.encoder_d1 = ...
        # self.encoder_d3 = ...
        # self.encoder_d5 = ...
        # self.encoder_to_semantic = ...
        
        # --- 新的: ResInc (x) 分支 ---
        # 我们希望它输出 [B, semantic_dim] 来匹配其他分支
        self.res_inc_extractor = Res_Inc_Net_base(num_classes=self.semantic_dim, include_top=True)


        # --- 最终组合层 (保留) ---
        self.total_semantic = nn.Sequential(
            # 输入维度为 x, y, z 三个分支的 semantic_dim 之和
            nn.Linear(self.semantic_dim * 3, self.semantic_dim, bias=False),
            nn.BatchNorm1d(self.semantic_dim),
            nn.ReLU(),
            nn.Linear(self.semantic_dim, self.semantic_dim, bias=False),
            nn.BatchNorm1d(self.semantic_dim),
            nn.ReLU()
        )
        self.semantic_to_classifier = nn.Sequential(
            nn.Linear(self.semantic_dim, self.num_class)
        )

    def forward(self, x_input): # 将 'x' 重命名为 'x_input' 以便区分
        
        # --- y 和 z 分支 (Transformer) ---
        # 这部分逻辑使用原始的 2D 输入 x_input，保持不变
        y = x_input.squeeze(1)                      # [B, 512, 512]
        z = y.permute(0, 2, 1)                      # [B, 512, 512]

        y = self.encoding_to_sa1(y)                 # [B, T, 64]
        z = self.encoding_to_sa2(z)                 # [B, W, 64]
        y = y + position_coding(y).to(self.device)
        z = z + position_coding(z).to(self.device)
        y = self.SA_1(y)                            # [B, T, 64]
        z = self.SA_2(z)                            # [B, W, 64]
        y = y.view(y.shape[0], -1)                  # [B, T*64]
        z = z.view(z.shape[0], -1)                  # [B, W*64]
        y = self.sa1_to_semantic(y)                 # [B, semantic_dim]
        z = self.sa2_to_semantic(z)                 # [B, semantic_dim]

        # --- x 分支 (ResInc) ---
        # 这是被替换的特征提取部分
        # 1. 将 2D 输入 [B, 1, 512, 512] 适配为 1D ResNet 的输入 [B, 1, 512]
        #    我们使用自适应平均池化压缩 H 维度
        # x_1d = F.adaptive_avg_pool2d(x_input, (1, self.input_size[1])) # [B, 1, 1, 512]
        # x_1d = x_1d.squeeze(2) # [B, 1, 512]

        x_1d = x_input.mean(dim=-1)

        # 2. 通过 ResInc 提取器
        x_features = self.res_inc_extractor(x_1d) # [B, semantic_dim]

        # --- 组合所有三个语义向量 ---
        semantic = torch.cat([x_features, y], dim=1)
        semantic = torch.cat([semantic, z], dim=1)
        semantic = self.total_semantic(semantic)  # [B, semantic_dim * 3] -> [B, semantic_dim]
        predict = self.semantic_to_classifier(semantic) # [B, semantic_dim] -> [B, num_class]

        return predict

# ======================================================
#  代码块 3: 测试脚本 (来自 S3R.py)
# ======================================================
if False:
    # device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    # # input_size=[T, W] -> [512, 512]
    # # semantic_dim (每个分支的特征维度) -> 128
    # # num_class (最终分类数) -> 10
    # model = NET(in_channels=1, input_size=[int(512), 512], semantic_dim=128, num_class=10, device=device).to(device)

    # # 你的输入要求: bs, 1, 512, 512
    # x = torch.randn(16, 1, 512, 512).to(device)  # batch_size=16
    # out = model(x)
    
    # # 你的输出要求: bs, 10
    # print(f"输入形状: {x.shape}")
    # print(f"输出形状: {out.shape}") # 应该打印: torch.Size([16, 10])

    # # (可选) 检查参数和计算量
    # try:
    #     from thop import profile
    #     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #     inputs = torch.randn(1, 1, 512, 512).to(device)
    #     flops, params = profile(model, (inputs,))
    #     print(f"FLOPs: {flops/1e9:.2f} G")
    #     print(f"Params: {params/1e6:.2f} M")
    # except ImportError:
    #     print("\n提示: 可以安装 'thop' (pip install thop) 来计算 FLOPs 和参数量。")

    from torchsummary import summary
    from thop import profile
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = NET(in_channels=1, input_size=[int(512), 512], semantic_dim=128, num_class=10, device=device).to(device)
    model = model.to(device)

    # summary(model, input_size=(1, 512))
    inputs = torch.randn(1, 1, 512, 512).to(device)
    flops, params = profile(model, (inputs,))
    print('flops: ', flops, 'params: ', params)