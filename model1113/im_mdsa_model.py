import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ST_E_Block(nn.Module):
    """
    SF-E 模块的基础构建块 (Spatial Feature Extractor Block)
    对应图 3 中的 "ST-E Block"：Conv -> Batch Norm -> ReLU
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1):
        super(ST_E_Block, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, 
                              stride=stride, padding=padding, dilation=dilation)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class SpatialFeatureExtractor(nn.Module):
    """
    空间特征提取器 (SF-E)
    基于图 3 (Spatial Feature Extractor) 的结构实现。
    注意：图 3 的图示和公式(10)有矛盾。
    - 公式(10) F_S = C_static ⊕ C_dynamic (⊕在论文中通常指代拼接)
    - 图 3 最后的 Marge 符号是 '+' (加法)
    - 加法会导致维度不匹配 (C_static: 2*dim, C_dynamic: dim)
    - 此处实现遵循公式(10)，采用拼接(concatenation)。
    """
    def __init__(self, in_channels, dim=64):
        super(SpatialFeatureExtractor, self).__init__()
        
        # Q, K1, K2, V 均由输入 X 生成
        self.q_conv = ST_E_Block(in_channels, dim, kernel_size=1)
        self.k1_conv = ST_E_Block(in_channels, dim, kernel_size=3, padding=1, dilation=1) # Conv(3), d=1
        self.k2_conv = ST_E_Block(in_channels, dim, kernel_size=3, padding=3, dilation=3) # Conv(3), d=3
        self.v_conv = ST_E_Block(in_channels, dim, kernel_size=1)
        
        # 用于计算注意力分数的卷积
        # 输入是 Q, K1, K2 的拼接，通道数为 3*dim
        self.attn_conv = ST_E_Block(dim * 3, dim, kernel_size=1)
        
        self.softmax = nn.Softmax(dim=1) # 论文未指定维度，假设为 channel-wise

    def forward(self, x):
        # 1. 生成 Q, K1, K2, V
        q = self.q_conv(x)   # (bs, dim, H, W)
        k1 = self.k1_conv(x) # (bs, dim, H, W)
        k2 = self.k2_conv(x) # (bs, dim, H, W)
        v = self.v_conv(x)   # (bs, dim, H, W)
        
        # 2. C_static (Eq. 6)
        c_static = torch.cat([k1, k2], dim=1) # (bs, 2*dim, H, W)
        
        # 3. C_dynamic (基于图 3 流程)
        # 3a. 拼接 Q, K1, K2 (对应图 3 "Concatenate")
        qk_cat = torch.cat([q, k1, k2], dim=1) # (bs, 3*dim, H, W)
        
        # 3b. 计算基础注意力分数
        attn_scores = self.attn_conv(qk_cat) # (bs, dim, H, W)
        
        # 3c. 对应图 3 中 Softmax 之前的加法
        # 论文原文[21]中，这里是 K_dynamic + Attn
        # 图 3 中是 V + Attn。我们遵循图 3。
        attn_plus_v = attn_scores + v
        
        # 3d. 对应图 3 "Softmax"
        attn_weights = self.softmax(attn_plus_v)
        
        # 3e. 对应图 3 中 Softmax 之后的乘法
        c_dynamic = attn_weights * v # (bs, dim, H, W)
        
        # 4. 融合 (Eq. 10)，使用拼接
        output = torch.cat([c_static, c_dynamic], dim=1) # (bs, 3*dim, H, W)
        
        return output

class TFF_E_Block(nn.Module):
    """
    TFF-E 模块的基础构建块 (Temporal-Frequency Feature Extractor Block)
    对应图 4 中的 "TFF-E Block"：Conv(n_i) -> Batch Norm -> ReLU
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(TFF_E_Block, self).__init__()
        # kernel_size 论文中为 (1, n_i)，这里实现为 (1, kernel_size)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
                              stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class PositionalEncoding(nn.Module):
    """
    标准正弦位置编码 (Eq. 12)
    """
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1) # (max_len, 1, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x 形状: (SeqLen, Batch, Dim)
        x = x + self.pe[:x.size(0), :]
        return x

class TemporalFrequencyFeatureExtractor(nn.Module):
    """
    时频特征提取器 (TFF-E)
    基于图 4 (Temporal-Frequency Feature Extractor) 的结构实现。
    
    为了处理 (512, 512) 的输入并使其在计算上可行（避免 512*512 的序列长度），
    我们将 Transformer 应用于 W 维度 (SeqLen=512)，特征维度为 (3*dim*H)。
    ...
    更新：一个更合理的假设是 Transformer 应用于 W 维度，同时 H 维度被视为特征的一部分。
    我们将 (bs, C, H, W) -> (bs, C*H, W) -> (W, bs, C*H) 作为 Transformer 输入。
    
    论文中 TFF-E Block 是并行的，但 kernel size 不明确 ("n_i=2 for all")
    我们假设是多尺度的 1D 卷积 (用 2D Conv 的 (1,k) 实现)
    """
    def __init__(self, in_channels, dim=64, n_head=4, num_encoder_layers=2, dim_feedforward=256):
        super(TemporalFrequencyFeatureExtractor, self).__init__()
        self.dim = dim
        
        # 1. 并行的 TFF-E 卷积块 (多尺度)
        # 我们假设 n_i=3, 5, 7 (奇数核，便于padding)
        self.block1 = TFF_E_Block(in_channels, dim, kernel_size=(1, 3), padding=(0, 1))
        self.block2 = TFF_E_Block(in_channels, dim, kernel_size=(1, 5), padding=(0, 2))
        self.block3 = TFF_E_Block(in_channels, dim, kernel_size=(1, 7), padding=(0, 3))
        
        self.fused_dim = dim * 3
        
        # 2. Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.fused_dim, 
            nhead=n_head, 
            dim_feedforward=dim_feedforward, 
            batch_first=True # 使用 (Batch, Seq, Dim)
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        # 3. 位置编码
        # 序列长度为 W = 512
        self.pos_encoder = PositionalEncoding(d_model=self.fused_dim, max_len=512)
        
        # 存储输入高度
        self.input_h = 512

    def forward(self, x):
        # x: (bs, 1, 512, 512)
        
        # 1. 并行卷积
        a1 = self.block1(x) # (bs, dim, 512, 512)
        a2 = self.block2(x) # (bs, dim, 512, 512)
        a3 = self.block3(x) # (bs, dim, 512, 512)
        
        # 2. 拼接 (图 4 "Concatenate")
        h = torch.cat([a1, a2, a3], dim=1) # (bs, 3*dim, 512, 512)
        
        # 3. 准备 Transformer 输入
        # (bs, 3*dim, H, W) -> (bs, H, 3*dim, W) -> (bs*H, 3*dim, W)
        bs, C, H, W = h.shape
        h_seq = h.permute(0, 2, 1, 3).reshape(bs * H, C, W)
        
        # (bs*H, C, W) -> (bs*H, W, C) (Batch, Seq, Dim)
        h_seq = h_seq.permute(0, 2, 1)
        
        # 4. 添加位置编码 (Eq. 12)
        # (W, bs*H, C)
        # h_seq_pos = self.pos_encoder(h_seq.permute(1, 0, 2))
        
        # 使用 batch_first=True 的 PosEncoding (更简单)
        # (bs*H, W, C)
        # 注意：PositionalEncoding 类需要修改以支持 batch_first
        # 简单起见，我们直接在 (W, B, C) 上操作
        h_seq_permuted = h_seq.permute(1, 0, 2) # (W, bs*H, C)
        h_seq_pos = self.pos_encoder(h_seq_permuted)
        
        # 5. Transformer 编码
        # TransformerEncoder 默认 (Seq, Batch, Dim)
        transformer_out = self.transformer_encoder(h_seq_pos) # (W, bs*H, C)
        
        # 6. 恢复形状
        # (W, bs*H, C) -> (bs*H, W, C)
        transformer_out = transformer_out.permute(1, 0, 2)
        
        # (bs*H, W, C) -> (bs*H, C, W) -> (bs, H, C, W) -> (bs, C, H, W)
        output = transformer_out.permute(0, 2, 1).reshape(bs, H, C, W).permute(0, 2, 1, 3)
        
        return output # (bs, 3*dim, 512, 512)

class CNNBlock(nn.Module):
    """
    分类器 CNN 的基础构建块 (图 5)
    Conv(3x3) -> Batch Norm -> ReLU
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class UAVClassifier(nn.Module):
    """
    UAV 分类器 (图 5)
    适配 (bs, C_in, 512, 512) 的输入
    保留了图 5 的 5 层结构和通道数 (8, 16, 64, 256, 256)
    使用 stride=2 的卷积来代替图 5 中不明确的下采样
    """
    def __init__(self, in_channels, num_classes=12):
        super(UAVClassifier, self).__init__()
        
        # 5层 CNN
        # 使用 stride=2 来匹配 (512 -> 256 -> 128 -> 64 -> 32 -> 16) 的下采样
        self.layer1 = CNNBlock(in_channels, 8, stride=2)
        self.layer2 = CNNBlock(8, 16, stride=2)
        self.layer3 = CNNBlock(16, 64, stride=2)
        self.layer4 = CNNBlock(64, 256, stride=2)
        self.layer5 = CNNBlock(256, 256, stride=2) 
        # 经过 5 次 stride=2，512 / (2^5) = 16
        # 输出 (bs, 256, 16, 16)
        
        # GAP (Global Average Pooling)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
        # Dense Layers
        self.fc1 = nn.Linear(256, 128)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        
        x = self.gap(x)
        x = torch.flatten(x, 1)
        
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x

class IM_MDSA(nn.Module):
    """
    完整的 IM-MDSA 模型
    (SF-E || TFF-E) -> Fusion -> Classifier
    """
    def __init__(self, in_channels=1, base_dim=64, num_classes=12, 
                 n_head=4, num_encoder_layers=2, dim_feedforward=256):
        super(IM_MDSA, self).__init__()
        
        # 1. 特征提取模块
        self.sf_e = SpatialFeatureExtractor(in_channels, dim=base_dim)
        # SF-E 输出 (bs, 3*base_dim, H, W)
        
        self.tff_e = TemporalFrequencyFeatureExtractor(
            in_channels, 
            dim=base_dim,
            n_head=n_head,
            num_encoder_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward
        )
        # TFF-E 输出 (bs, 3*base_dim, H, W)
        
        fused_channels = (3 * base_dim) + (3 * base_dim) # 6 * base_dim = 384
        
        # 2. 分类器模块
        # (论文图 5 中第一层是 4 通道，我们用一个 1x1 卷积来对齐)
        self.entry_conv = CNNBlock(fused_channels, 4, kernel_size=1, padding=0)
        
        # 输入通道为 4
        self.classifier = UAVClassifier(in_channels=4, num_classes=num_classes)
        

    def forward(self, x):
        # x: (bs, 1, 512, 512)
        
        # 1. 并行特征提取
        f_s = self.sf_e(x)   # (bs, 192, 512, 512)
        f_tf = self.tff_e(x) # (bs, 192, 512, 512)
        
        # 2. 特征融合 (Eq. 14)
        f_0 = torch.cat([f_s, f_tf], dim=1) # (bs, 384, 512, 512)
        
        # 3. 分类
        # 适配图 5 的 4 通道输入
        f_in = self.entry_conv(f_0) # (bs, 4, 512, 512)
        output = self.classifier(f_in)
        
        return output

# --- 示例用法 ---
if __name__ == "__main__":
    # 1. 检查模型是否能处理 (bs, 1, 512, 512) 的输入
    batch_size = 1
    num_classes = 12 # 根据论文 Table II
    
    # 使用较小的参数以加快测试速度
    # 论文声称模型轻量级 (0.52M params)，
    # 完整的 Transformer 可能会使参数量变得很大。
    # 我们可以减少 base_dim 和 transformer 的参数
    
    model = IM_MDSA(
        in_channels=1,
        base_dim=32,       # 减少基础维度 (原 64)
        num_classes=num_classes,
        n_head=4,
        num_encoder_layers=1, # 减少层数 (原 2)
        dim_feedforward=128 # 减少前馈维度 (原 256)
    )
    
    # 打印模型参数量
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总参数量: {total_params / 1e6:.2f} M")
    
    # 创建一个随机输入
    dummy_input = torch.randn(batch_size, 1, 512, 512)
    
    # 前向传播
    try:
        output = model(dummy_input)
        print(f"输入形状: {dummy_input.shape}")
        print(f"输出形状: {output.shape}")
        assert output.shape == (batch_size, num_classes)
        print("模型构建成功并通过形状检查！")
        
    except Exception as e:
        print(f"模型构建失败: {e}")
        import traceback
        traceback.print_exc()

    # 打印模型结构 (可选)
    # print(model)