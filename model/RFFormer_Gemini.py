import torch
import torch.nn as nn

class FreqEncoder(nn.Module):
    """
    频率轴特征提取器 (1D CNN)
    输入: (N, 1, 512) - N是 (bs * T), 1是通道, 512是频率点
    输出: (N, embed_dim) - 压缩后的特征
    """
    def __init__(self, embed_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            # (N, 1, 512) -> (N, 32, 256)
            nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            # (N, 32, 256) -> (N, 64, 128)
            nn.Conv1d(32, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            # (N, 64, 128) -> (N, 128, 64)
            nn.Conv1d(64, 128, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            # (N, 128, 64) -> (N, embed_dim, 32)
            nn.Conv1d(128, embed_dim, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True),
            
            # 全局平均池化
            # (N, embed_dim, 32) -> (N, embed_dim, 1)
            nn.AdaptiveAvgPool1d(1),
            # (N, embed_dim, 1) -> (N, embed_dim)
            nn.Flatten()
        )

    def forward(self, x):
        return self.encoder(x)

class RFFormer(nn.Module):
    """
    RFformer: 结合了CNN频域编码器和Transformer时域建模器
    
    输入: (bs, 1, 512, 512)  (bs, C, Freq, Time)
    输出: (bs, 10)
    """
    def __init__(self, num_classes=10, seq_len=512, embed_dim=128, num_layers=4, num_heads=4):
        super().__init__()
        self.embed_dim = embed_dim
        
        # 1. 频率轴特征提取器
        self.freq_encoder = FreqEncoder(embed_dim=embed_dim)
        
        # 2. 时序Transformer
        # TransformerEncoderLayer需要 (SeqLen, Batch, EmbedDim) 或 (Batch, SeqLen, EmbedDim)
        # 我们使用 batch_first=True
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=embed_dim * 4, 
            dropout=0.1,
            batch_first=True  # 关键: 输入形状为 (bs, SeqLen, EmbedDim)
        )
        self.temporal_transformer = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )
        
        # 3. 可学习的位置编码 (因为时序是关键)
        # 我们有 seq_len (512) 个token
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, embed_dim))
        
        # 4. 分类头
        # 我们将所有时间步的输出取平均，而不是使用[CLS]Token，这更轻量
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )

        self.classifier1 = nn.Linear(128 * self.embed_dim, num_classes)

    def forward(self, x):
        # 输入 x: (bs, 1, 512, 512)  (bs, C, Freq, Time)
        bs, _, F, T = x.shape
        
        # 准备数据给FreqEncoder (1D CNN)
        # 1D CNN需要 (N, C_in, L_in)
        # 我们把 Freq (F=512) 作为 L_in
        # 我们把 (bs * T) 作为 N
        
        # (bs, 1, F, T) -> (bs, F, T)
        x = x.squeeze(1)
        # (bs, F, T) -> (bs, T, F)
        x = x.permute(0, 2, 1)
        # (bs, T, F) -> (bs * T, F)
        x = x.reshape(bs * T, F)
        # (bs * T, F) -> (bs * T, 1, F)  (N, C_in, L_in)
        x = x.unsqueeze(1)
        
        # 阶段一：频率轴编码
        # (bs * T, 1, 512) -> (bs * T, embed_dim)
        # 这里的N = bs * T
        x_tokens = self.freq_encoder(x)
        
        # 准备数据给Transformer
        # (bs * T, embed_dim) -> (bs, T, embed_dim)
        # 这正是 (bs, SeqLen, EmbedDim) 的格式
        x_seq = x_tokens.reshape(bs, T, self.embed_dim)
        
        x_seq = x_seq.reshape(bs, T * self.embed_dim)
        output = self.classifier1(x_seq)
        return output

        # 添加位置编码
        x_seq = x_seq + self.pos_embed
        
        # 阶段二：时间轴时序建模
        # (bs, T, embed_dim) -> (bs, T, embed_dim)
        x_transformed = self.temporal_transformer(x_seq)
        
        # 阶段三：分类
        # 使用全局平均池化 (在时间维度上)
        # (bs, T, embed_dim) -> (bs, embed_dim)
        x_pooled = x_transformed.mean(dim=1)
        
        # (bs, embed_dim) -> (bs, 10)
        output = self.classifier(x_pooled)
        
        return output
    
if False:
    # 模拟输入数据
    # (batch_size, channels, freq_bins, time_steps)
    dummy_input = torch.randn(16, 1, 128, 128)
    
    print("--- 测试 RFformer (方案一: CNN+Transformer) ---")
    # 为了轻量化，我们使用 embed_dim=128, 4层, 4个头
    model_v1 = RFFormer(
        num_classes=10, 
        seq_len=128, 
        embed_dim=128, 
        num_layers=4, 
        num_heads=4
    )
    
    output_v1 = model_v1(dummy_input)
    print(f"输入形状: {dummy_input.shape}")
    print(f"输出形状 (V1): {output_v1.shape}") # 应该为 (16, 10)