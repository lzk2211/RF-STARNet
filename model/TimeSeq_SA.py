from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionalEncoding(nn.Module):
    """标准正弦-余弦位置编码，batch_first=(B,L,D)。"""
    def __init__(self, d_model: int, max_len: int = 2048, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)  # (1,max_len,d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,L,D)
        L = x.size(1)
        x = x + self.pe[:, :L, :]
        return self.dropout(x)


class RelativePositionBias(nn.Module):
    """可学习的相对位置偏置，类似 T5，相对距离裁剪到 [-max_dist+1, max_dist-1]。"""
    def __init__(self, num_heads: int, max_dist: int = 128):
        super().__init__()
        self.num_heads = num_heads
        self.max_dist = max_dist
        self.num_buckets = 2 * max_dist - 1
        self.bias_table = nn.Parameter(torch.zeros(self.num_buckets, num_heads))
        nn.init.trunc_normal_(self.bias_table, std=0.02)

    def forward(self, L: int) -> torch.Tensor:
        # 生成 (L,L) 的相对距离索引，映射到 [0, num_buckets-1]
        pos = torch.arange(L)
        rel = pos[None, :] - pos[:, None]  # (L,L) in [-L+1, L-1]
        rel = rel.clamp(min=-(self.max_dist - 1), max=(self.max_dist - 1))
        rel = rel + (self.max_dist - 1)  # shift to [0, num_buckets-1]
        bias = self.bias_table[rel]       # (L,L,H)
        return bias.permute(2, 0, 1).contiguous()  # (H,L,L)


class MultiHeadSelfAttention(nn.Module):
    """轻量化自注意力，支持相对位置偏置。"""
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.0, use_rel_pos_bias: bool = True, max_len: int = 128):
        super().__init__()
        assert d_model % nhead == 0, "d_model 必须能被 nhead 整除"
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=True)
        self.proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

        self.use_rel_pos_bias = use_rel_pos_bias
        self.rpb = RelativePositionBias(nhead, max_len) if use_rel_pos_bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,L,D)
        B, L, D = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.nhead, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # (B,H,L,dh)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B,H,L,L)
        if self.use_rel_pos_bias and self.rpb is not None:
            bias = self.rpb(L)  # (H,L,L)
            attn = attn + bias.unsqueeze(0)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = attn @ v  # (B,H,L,dh)
        out = out.transpose(1, 2).reshape(B, L, D)
        out = self.proj_drop(self.proj(out))
        return out


class FeedForward(nn.Module):
    def __init__(self, d_model: int, dim_ff: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerEncoderBlock(nn.Module):
    """Pre-LN Transformer 编码块。"""
    def __init__(self, d_model: int, nhead: int, dim_ff: int, dropout: float = 0.0, use_rel_pos_bias: bool = True, max_len: int = 128):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, nhead, dropout, use_rel_pos_bias, max_len)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, dim_ff, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


class TimeSeqSA(nn.Module):
    """
    轻量级时序自注意力模型（算力接近 Res_Inc_Net）：
    - 输入: (B, 1, 128)
    - 输出（优先级从高到低）:
        * include_top=True: 分类 logits (B, num_classes)
        * out_dim 设置: (B, out_dim)
        * 否则: (B, 1, 128) 的序列输出
    """
    def __init__(
        self,
        d_model: int = 32,
        nhead: int = 4,
        num_layers: int = 1,
        ff_ratio: int = 2,
        dropout: float = 0.1,
        max_len: int = 128,
        use_rel_pos_bias: bool = True,
        out_dim: int | None = None,
        pool_stride: int = 2,          # 序列下采样步长(=1时不下采样)
        num_classes: int = 1000,       # 新增：分类类别数，默认与 Res_Inc_Net 一致
        include_top: bool = True,      # 新增：是否启用分类头
    ):
        super().__init__()
        self.max_len = max_len
        self.include_top = include_top
        self.num_classes = num_classes

        self.proj_in = nn.Linear(1, d_model)

        # 可选序列下采样
        self.pool_stride = max(1, int(pool_stride))
        self.seq_down = None if self.pool_stride == 1 else nn.AvgPool1d(kernel_size=self.pool_stride, stride=self.pool_stride)

        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len=max_len, dropout=dropout)

        dim_ff = d_model * ff_ratio
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(d_model, nhead, dim_ff, dropout, use_rel_pos_bias, max_len)
            for _ in range(num_layers)
        ])

        # 头部定义（优先级：分类头 > 向量头 > 序列重建头）
        if self.include_top:
            self.proj_out = None
            self.norm = nn.LayerNorm(d_model)
            self.dropout_head = nn.Dropout(dropout)
            self.cls = nn.Linear(d_model, num_classes)
        elif out_dim is not None:
            self.proj_out = None
            self.norm = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, out_dim)
        else:
            self.proj_out = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 1, 128)
        返回:
          - include_top=True: (B, num_classes)
          - out_dim is not None: (B, out_dim)
          - 否则: (B, 1, 128)
        """
        assert x.dim() == 3 and x.size(1) == 1, "输入形状应为 (B,1,128)"
        orig_len = x.size(-1)

        x = x.transpose(1, 2)      # (B,128,1)
        x = self.proj_in(x)        # (B,128,D)

        # 下采样以降低注意力算力
        if self.seq_down is not None:
            x = x.transpose(1, 2)                  # (B,D,L)
            x = self.seq_down(x)                   # (B,D,L')
            x = x.transpose(1, 2).contiguous()     # (B,L',D)

        # 位置编码放在下采样之后
        x = self.pos_enc(x)

        for blk in self.blocks:
            x = blk(x)

        # 分类头优先生效
        if self.include_top:
            x = x.mean(dim=1)                      # (B,D)
            x = self.norm(x)
            x = self.dropout_head(x)
            return self.cls(x)                     # (B,num_classes)

        # 嵌入向量输出头
        if hasattr(self, "head"):
            z = x.mean(dim=1)                      # (B,D)
            z = self.norm(z)
            return self.head(z)                    # (B,out_dim)

        # 序列重建头
        y = self.proj_out(x).transpose(1, 2)       # (B,1,L')
        if y.size(-1) != orig_len:
            y = F.interpolate(y, size=orig_len, mode="linear", align_corners=False)
        return y


# 简单自测
if __name__ == "__main__":
    # 分类模式
    model = TimeSeqSA(num_classes=25, include_top=True)
    x = torch.randn(8, 1, 128)
    y = model(x)
    print("cls out:", y.shape)  # (8,25)
    # 序列输出模式
    model_seq = TimeSeqSA(include_top=False, out_dim=None)
    y_seq = model_seq(x)
    print("seq out:", y_seq.shape)  # (8,1,128)
    # 参数量
    total = sum(p.numel() for p in model.parameters())
    print("params:", total)
