import torch
import torch.nn as nn
import torch.nn.functional as F
# from thop import profile # 取消注释以进行性能测试


# ============================================================
# 🧩 1D Ultra-Stable NoiseGateConv
# ============================================================
class NoiseGateConv1d(nn.Module):
    def __init__(self, channels):
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
# 🧩 1D NGConvBlock：Conv1d → BN1d → ReLU → NoiseGateConv1d
# ============================================================
class NGConvBlock1d(nn.Module):
    def __init__(self, in_ch, out_ch, dilation):
        super().__init__()
        self.conv = nn.Conv1d(
            in_ch, out_ch, 3,
            padding=dilation,
            dilation=dilation,
            bias=False
        )
        self.bn = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.ng = NoiseGateConv1d(out_ch)

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        x = self.ng(x)
        return x


# ============================================================
# 🔥 最终优化版 1D TinyTransformer
# ============================================================
class S3R_1D_TinyTransformer(nn.Module):
    def __init__(
        self,
        in_channels=1,
        num_classes=10,
        semantic_dim=128,
        patch_embed_dim=128,
        nhead=4,
        num_layers=1,
    ):
        super().__init__()

        # ----------------------------------------------------
        # 1. 三路 1D Multi-Scale Encoder（1,3,5 膨胀）
        # ----------------------------------------------------
        self.enc1 = self._make_encoder_1d(in_channels, 1)
        self.enc3 = self._make_encoder_1d(in_channels, 3)
        self.enc5 = self._make_encoder_1d(in_channels, 5)

        # 合并 3 路输出：64×3 = 192
        self.proj = nn.Conv1d(192, patch_embed_dim, 1, bias=False)

        # ----------------------------------------------------
        # 2. 1D Positional Embedding（基础长度 32）
        # ----------------------------------------------------
        # 输入 512，经过 4 次 MaxPool1d(2)
        # 512 -> 256 -> 128 -> 64 -> 32
        self.base_len = 32
        self.register_parameter(
            "pos_embed",
            nn.Parameter(torch.randn(1, self.base_len, patch_embed_dim))
        )

        # ----------------------------------------------------
        # 3. Transformer Encoder
        # ----------------------------------------------------
        layer = nn.TransformerEncoderLayer(
            d_model=patch_embed_dim,
            nhead=nhead,
            dim_feedforward=patch_embed_dim * 2,
            dropout=0.05,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)

        # ----------------------------------------------------
        # 4. Semantic Head
        # ----------------------------------------------------
        self.semantic_head = nn.Sequential(
            nn.Linear(patch_embed_dim, semantic_dim * 2),
            nn.LayerNorm(semantic_dim * 2),
            nn.GELU(),
            nn.Linear(semantic_dim * 2, semantic_dim),
            nn.LayerNorm(semantic_dim),
            nn.GELU(),
        )
        self.classifier = nn.Linear(semantic_dim, num_classes)


    # ----------------------------------------------------
    # 1D CNN encoder（带 NoiseGate）
    # ----------------------------------------------------
    def _make_encoder_1d(self, in_ch, dilation):
        return nn.Sequential(
            NGConvBlock1d(in_ch,  8, dilation), nn.MaxPool1d(2),  # 512 -> 256
            NGConvBlock1d(8,    16, dilation), nn.MaxPool1d(2),  # 256 -> 128
            NGConvBlock1d(16,   32, dilation), nn.MaxPool1d(2),  # 128 -> 64
            NGConvBlock1d(32,   48, dilation), nn.MaxPool1d(2),  # 64 -> 32
            NGConvBlock1d(48,   64, dilation),                   # 32
        )


    # ----------------------------------------------------
    # Forward
    # ----------------------------------------------------
    def forward(self, x):
        # ----------------------------------------------------
        # 🔥 核心修改：将 [B, 1, 512, 512] 降维到 [B, 1, 512]
        # ----------------------------------------------------
        x = x.mean(dim=-1)
        # ----------------------------------------------------

        # 1) multi-scale 1D CNN
        e1 = self.enc1(x) # [B, 64, 32]
        e3 = self.enc3(x) # [B, 64, 32]
        e5 = self.enc5(x) # [B, 64, 32]

        feat = torch.cat([e1, e3, e5], dim=1)  # [B, 192, 32]
        feat = self.proj(feat)                 # [B, 128, 32]

        B, C, L = feat.shape
        token_len = L

        # 2) flatten tokens (将 [B, C, L] 转为 [B, L, C])
        tokens = feat.transpose(1, 2)   # [B, 32, 128]

        # ----------------------------------------------------
        # 🔥 动态 1D Positional Encoding
        # ----------------------------------------------------
        if token_len != self.base_len:
            # 1D 线性插值
            # [1, base_len, C] -> [1, C, base_len]
            pos = self.pos_embed.transpose(1, 2)
            pos = F.interpolate(
                pos, size=L,
                mode="linear", align_corners=False
            )
            # [1, C, L] -> [1, L, C]
            pos = pos.transpose(1, 2)
        else:
            pos = self.pos_embed

        tokens = tokens + pos
        # ----------------------------------------------------

        # transformer
        tokens = self.transformer(tokens)

        # global semantic feature
        global_feat = tokens.mean(dim=1)

        semantic = self.semantic_head(global_feat)
        logits = self.classifier(semantic)

        return logits#, semantic


if __name__ == "__main__":
	# from torchsummary import summary
	from thop import profile
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	model = S3R_1D_TinyTransformer(num_classes=10) # <-- 使用 1D 版本
	model = model.to(device)

	# summary(model, input_size=(1, 512)) # 1D summary 不太好用
	
    # 输入仍然是 [B, 1, 512, 512]，模型内部会处理降维
	inputs = torch.randn(1, 1, 512, 512).to(device) 
	
	flops, params = profile(model, (inputs,), verbose=False)
	print('flops: ', flops, 'params: ', params)