import torch
import torch.nn as nn
from torch.nn import functional as F

class AsymmetricPatchEmbed(nn.Module):
    """
    非对称的图像Patch嵌入
    使用一个Conv2d来实现
    """
    def __init__(self, in_chans=1, embed_dim=128, patch_size_f=64, patch_size_t=8):
        super().__init__()
        self.patch_size = (patch_size_f, patch_size_t)
        
        # 使用一个Conv2d来做Patching和Embedding
        self.proj = nn.Conv2d(
            in_chans, 
            embed_dim, 
            kernel_size=self.patch_size, 
            stride=self.patch_size
        )

    def forward(self, x):
        # x: (bs, 1, 512, 512) (bs, C, F, T)
        # (bs, 1, 512, 512) -> (bs, embed_dim, 512/64, 512/8)
        # (bs, embed_dim, 8, 64)
        x = self.proj(x)
        
        # (bs, C, H, W) -> (bs, C, H*W) -> (bs, H*W, C)
        # (bs, embed_dim, 8, 64) -> (bs, embed_dim, 512)
        x = x.flatten(2)
        # (bs, embed_dim, 512) -> (bs, 512, embed_dim)
        x = x.transpose(1, 2)
        return x

class RFformer_v2(nn.Module):
    """
    RFformer V2: 使用非对称Patch的Vision Transformer
    
    输入: (bs, 1, 512, 512)  (bs, C, Freq, Time)
    输出: (bs, 10)
    """
    def __init__(self, num_classes=10, img_size_f=512, img_size_t=512, 
                 patch_size_f=64, patch_size_t=8, embed_dim=128, 
                 num_layers=4, num_heads=4):
        super().__init__()
        
        # 1. 非对称Patch嵌入
        self.patch_embed = AsymmetricPatchEmbed(
            in_chans=1,
            embed_dim=embed_dim,
            patch_size_f=patch_size_f,
            patch_size_t=patch_size_t
        )
        
        # 计算Token数量
        num_patches_f = img_size_f // patch_size_f  # 512 / 64 = 8
        num_patches_t = img_size_t // patch_size_t  # 512 / 8 = 64
        num_patches = num_patches_f * num_patches_t # 8 * 64 = 512
        
        # 2. [CLS] Token 和 位置编码
        # ViT标准做法
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))
        
        # 3. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )
        
        # 4. 分类头
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, x):
        # x: (bs, 1, 512, 512)
        bs = x.shape[0]
        
        # 1. Patch 嵌入
        # (bs, 1, 512, 512) -> (bs, 512, embed_dim) (bs, NumPatches, EmbedDim)
        x = self.patch_embed(x)
        
        # 2. 添加 [CLS] Token
        # (1, 1, embed_dim) -> (bs, 1, embed_dim)
        cls_tokens = self.cls_token.expand(bs, -1, -1)
        # (bs, 512, embed_dim) -> (bs, 513, embed_dim)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # 3. 添加位置编码
        x = x + self.pos_embed
        
        # 4. Transformer
        # (bs, 513, embed_dim) -> (bs, 513, embed_dim)
        x = self.transformer_encoder(x)
        
        # 5. 分类
        # 只取 [CLS] Token 的输出
        # (bs, 513, embed_dim) -> (bs, embed_dim)
        cls_token_output = x[:, 0]
        
        # (bs, embed_dim) -> (bs, 10)
        output = self.head(cls_token_output)
        
        return output