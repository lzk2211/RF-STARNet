import torch
import torch.nn as nn
import torch.nn.functional as F



# ============================================================
# 🧩 Ultra-Stable NoiseGateConv（可学习，永不发散）
# ============================================================
class NoiseGateConv(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # 原始可学习参数（softplus 后总 >= 0）
        self.k_raw = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        # k 始终为正且可控
        k = F.softplus(self.k_raw)

        mean = self.avgpool(x)
        sigma = (x - mean).abs().mean(dim=(2,3), keepdim=True)

        # 稳定 gate：避免 1/sigma 形式导致爆炸
        gate = 1.0 / (1.0 + k * sigma)

        # 限制 gate 范围，避免关死
        gate = torch.clamp(gate, 0.3, 1.0)

        return x * gate


# ============================================================
# 🧩 NGConvBlock：Conv → BN → ReLU → NoiseGateConv
# ============================================================
class NGConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dilation):
        super().__init__()
        self.conv = nn.Conv2d(
            in_ch, out_ch, 3,
            padding=dilation,
            dilation=dilation,
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.ng = NoiseGateConv(out_ch)

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        x = self.ng(x)
        return x


# ============================================================
# 🔥 最终优化版 TinyTransformer
# ============================================================
class S3RImageNet_TinyTransformer(nn.Module):
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
        # 1. 三路 Multi-Scale Encoder（1,3,5 膨胀）
        # ----------------------------------------------------
        self.enc1 = self._make_encoder(in_channels, 1)
        self.enc3 = self._make_encoder(in_channels, 3)
        self.enc5 = self._make_encoder(in_channels, 5)

        # 合并 3 路输出：64×3 = 192
        self.proj = nn.Conv2d(192, patch_embed_dim, 1, bias=False)

        # ----------------------------------------------------
        # 2. Positional Embedding（基础 7×7）
        # ----------------------------------------------------
        self.base_hw = 7   # 来自 224→112→56→28→14→7
        self.register_parameter(
            "pos_embed",
            nn.Parameter(torch.randn(1, self.base_hw*self.base_hw, patch_embed_dim))
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
    # CNN encoder（带 NoiseGate）
    # ----------------------------------------------------
    def _make_encoder(self, in_ch, dilation):
        return nn.Sequential(
            NGConvBlock(in_ch,  8, dilation), nn.MaxPool2d(2),
            NGConvBlock(8,    16, dilation), nn.MaxPool2d(2),
            NGConvBlock(16,   32, dilation), nn.MaxPool2d(2),
            NGConvBlock(32,   48, dilation), nn.MaxPool2d(2),
            NGConvBlock(48,   64, dilation),
        )


    # ----------------------------------------------------
    # Forward
    # ----------------------------------------------------
    def forward(self, x):

        # 1) multi-scale CNN
        e1 = self.enc1(x)
        e3 = self.enc3(x)
        e5 = self.enc5(x)

        feat = torch.cat([e1, e3, e5], dim=1)  # [B,192,H,W]
        feat = self.proj(feat)                 # [B,128,H,W]

        B, C, H, W = feat.shape
        token_len = H * W

        # 2) flatten tokens
        tokens = feat.flatten(2).transpose(1, 2)   # [B,HW,128]

        # ----------------------------------------------------
        # 🔥 动态 Positional Encoding（无论 HW=49 或 196 都兼容）
        # ----------------------------------------------------
        if token_len != self.base_hw * self.base_hw:
            # reshape → interpolate → reshape back
            pos = self.pos_embed.reshape(1, self.base_hw, self.base_hw, -1)
            pos = pos.permute(0, 3, 1, 2)  # [1,128,7,7]

            pos = F.interpolate(
                pos, size=(H, W),
                mode="bilinear", align_corners=False
            )
            pos = pos.permute(0, 2, 3, 1).reshape(1, H*W, -1)
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

	model = S3RImageNet_TinyTransformer(num_classes=10)
	model = model.to(device)

	# summary(model, input_size=(1, 512))
	inputs = torch.randn(1, 1, 512, 512).to(device)
	flops, params = profile(model, (inputs,), verbose=False)
	print('flops: ', flops, 'params: ', params)
