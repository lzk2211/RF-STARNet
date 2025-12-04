import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

class RelativePositionBias(nn.Module):
    """
    相对位置偏置（Relative Position Bias）。

    作用：为注意力打分矩阵提供依赖“相对距离”的可学习偏置，使模型具备时序/相对位置信息感知。

    特性：
    - 支持非方阵 (q_len, k_len)；
    - 参数按参考长度 seq_len 构建，范围覆盖相对距离 d ∈ [-(S-1), S-1]，共 2S-1 个；
    - 返回形状为 (num_heads, q_len, k_len)，可与 (B, q_len, k_len) 广播相加。
    """
    def __init__(self, seq_len: int, num_heads: int = 1):
        super().__init__()
        self.seq_len = int(seq_len)
        self.num_heads = int(num_heads)

        # 支持距离范围 [-S+1, S-1]
        self.num_rel_pos = 2 * self.seq_len - 1

        # 每个相对距离对应一个偏置值
        self.relative_bias = nn.Parameter(torch.zeros(self.num_heads, self.num_rel_pos))
        nn.init.trunc_normal_(self.relative_bias, std=0.02)

        # 预计算正方情形 (S, S) 的索引表，非方阵时动态构造
        coords = torch.arange(self.seq_len)
        rel_pos = coords[None, :] - coords[:, None]  # i - j
        rel_pos += self.seq_len - 1  # shift to [0, 2S-2]
        self.register_buffer("rel_pos_index", rel_pos)

    def forward(self, q_len: int, k_len: Optional[int] = None) -> torch.Tensor:
        """
        返回偏置矩阵，形状 (num_heads, q_len, k_len)。
        若 k_len 省略或等于 q_len，则退化为方阵 (q_len, q_len)。
        """
        if k_len is None:
            k_len = q_len

        q_len = int(q_len)
        k_len = int(k_len)

        # 快路径：与参考长度一致且为方阵
        if q_len == k_len == self.seq_len:
            idx = self.rel_pos_index  # (S, S)
            return self.relative_bias[:, idx]  # (H, S, S)

        device = self.relative_bias.device

        # 将查询/键位置线性映射到参考序列 [0, S-1]
        S = self.seq_len
        q_pos_ref = torch.round(torch.linspace(0, S - 1, steps=q_len, device=device))  # (q_len,)
        k_pos_ref = torch.round(torch.linspace(0, S - 1, steps=k_len, device=device))  # (k_len,)

        # 计算整数相对位移并映射到 [0, 2S-2]
        rel = q_pos_ref[:, None] - k_pos_ref[None, :]  # (q_len, k_len)
        rel = rel.clamp(min=-(S - 1), max=(S - 1)).to(torch.long)
        idx = (rel + (S - 1)).to(torch.long)  # (q_len, k_len)

        # 索引查表 -> (H, q_len, k_len)
        return self.relative_bias[:, idx]
    
class GroupedProjectedAttentionRF(nn.Module):
    """
    分组注意力（无投影版）。输入形状 (B, C, F, T)，内部按组将 F 维切分为 G 份，
    每组独立计算 self-attention，并在时间维度上加入“相对位置偏置”。
    输出形状 (B, F, T)。
    """

    def __init__(self,
                 freq_bins: int,
                 att_dim: int,
                 groups: int,
                 seq_len: int,
                 use_pos: bool = True,
                 dropout: float = 0.0):
        super().__init__()
        assert freq_bins % groups == 0, "freq_bins must be divisible by groups"
        self.freq_bins = freq_bins
        self.att_dim = att_dim
        self.groups = groups
        self.gdim = freq_bins // groups
        self.seq_len = seq_len
        self.scale = 1.0 / math.sqrt(att_dim)
        self.use_pos = use_pos

        # per-group linear projections
        self.qs = nn.ModuleList([nn.Linear(self.gdim, att_dim, bias=False) for _ in range(groups)])
        self.ks = nn.ModuleList([nn.Linear(self.gdim, att_dim, bias=False) for _ in range(groups)])
        self.vs = nn.ModuleList([nn.Linear(self.gdim, self.gdim, bias=False) for _ in range(groups)])
        self.os = nn.ModuleList([nn.Linear(self.gdim, self.gdim, bias=False) for _ in range(groups)])

        # 相对位置偏置：按组（num_heads = groups）各自一份
        self.rel_pos_bias = RelativePositionBias(seq_len=seq_len, num_heads=groups)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.layer_norm = nn.LayerNorm(freq_bins)
        # 位置编码模块暂不启用，如需可外部接入

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, F, T)
        returns: out (B, F, T)
        """

        B, C, F, T = x.shape
        x = x.squeeze(1)  # (B, F, T)
        assert F == self.freq_bins, f"Input freq dim {F} must equal freq_bins {self.freq_bins}"

        x_t = x
        x_t = x_t.permute(0, 2, 1)  # (B, T, F)
        x_t = self.layer_norm(x_t)  # normalize over F

        B, T2, D = x_t.shape  # T2 == 原始 T, D == F
        G = self.groups
        gdim = self.gdim

        xg = x_t.view(B, T2, G, gdim).permute(0, 2, 1, 3)  # (B, G, T, gdim)

        outs = []
        # 预先计算所有组的相对位置偏置，减少重复构造
        bias_all = self.rel_pos_bias(T2, T2).to(x_t.device)  # (G, T, T)
        for g in range(G):
            xg_g = xg[:, g]  # (B, T, gdim)
            q = self.qs[g](xg_g)  # (B, T, att_dim)
            k = self.ks[g](xg_g)  # (B, T, att_dim)
            v = self.vs[g](xg_g)  # (B, T, gdim)

            att = torch.matmul(q, k.transpose(1, 2)) * self.scale

            bias_g = bias_all[g].unsqueeze(0)  # (1, T, T)
            att = att + bias_g  # (B, T, T) + (1, T, T)

            att = torch.softmax(att, dim=-1)
            att = self.dropout(att)

            out_g = torch.matmul(att, v)
            out_g = self.os[g](out_g)  # (B, T, gdim)
            outs.append(out_g)

        out = torch.cat(outs, dim=-1)  # (B, T, F)
        out = out.permute(0, 2, 1).contiguous()  # (B, F, T)
        return out


class RFProjectedAttentionClassifier(nn.Module):
    """
    分类器封装：前端使用 GroupedProjectedAttentionRF 获得 (B, F, T)，
    再在时间维度做平均池化得到 (B, F)，接多层感知机输出分类 logits。
    """

    def __init__(self,
                 freq_bins: int,
                 seq_len: int,
                 num_classes: int = 25,
                 groups: int = 8,
                 att_dim: int = 32,
                 hidden_dim: Optional[int] = None,
                 use_pos: bool = True,
                 dropout: float = 0.1):
        super().__init__()
        self.att = GroupedProjectedAttentionRF(
            freq_bins=freq_bins,
            att_dim=att_dim,
            groups=groups,
            seq_len=seq_len,
            use_pos=use_pos,
            dropout=dropout
        )
        
        if hidden_dim is None:
            hidden_dim = max(128, freq_bins // 2)

        self.fc = nn.Sequential(
            nn.Linear(freq_bins, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, F, T)  (例如 (B, 1, 512, 512))
        returns logits: (B, num_classes)
        """
        out = self.att(x)  # (B, F, T)
        pooled = out.mean(dim=-1)  # (B, F)
        logits = self.fc(pooled)   # (B, num_classes)
        return logits




def T_att_nano(freq_bins: int, seq_len: int, num_classes: int, **kwargs): # <-- 【改动】include_top 成为显式参数

    config = {
        'groups': 1,
        'att_dim': 2,
        'hidden_dim': 4,
        'dropout': 0.1
    }
    
    # 允许 kwargs 覆盖默认配置 (例如 dropout=0.0)
    config.update(kwargs) 
    
    return RFProjectedAttentionClassifier(
        freq_bins=freq_bins,
        seq_len=seq_len,
        num_classes=num_classes,
        **config
    )


def T_att_mini(freq_bins: int, seq_len: int, num_classes: int, **kwargs): # <-- 【改动】include_top 成为显式参数

    config = {
        'groups': 1,
        'att_dim': 4,
        'hidden_dim': 8,
        'dropout': 0.1
    }
    
    # 允许 kwargs 覆盖默认配置 (例如 dropout=0.0)
    config.update(kwargs) 
    
    return RFProjectedAttentionClassifier(
        freq_bins=freq_bins,
        seq_len=seq_len,
        num_classes=num_classes,
        **config
    )

def T_att_tiny(freq_bins: int, seq_len: int, num_classes: int, **kwargs): # <-- 【改动】include_top 成为显式参数
    """
    创建 'tiny' 版本：G=64, att_dim=4, hidden_dim=16。
    """
    config = {
        'groups': 1,
        'att_dim': 4,
        'hidden_dim': 16,
        'dropout': 0.1
    }
    
    # 允许 kwargs 覆盖默认配置 (例如 dropout=0.0)
    config.update(kwargs) 
    
    return RFProjectedAttentionClassifier(
        freq_bins=freq_bins,
        seq_len=seq_len,
        num_classes=num_classes,
        **config
    )

def T_att_small(freq_bins: int, seq_len: int, num_classes: int, **kwargs): # <-- 【改动】include_top 成为显式参数
    """
    创建 'small' 版本：G=8, att_dim=16，hidden_dim=128。
    """
    config = {
        'groups': 1,      
        'att_dim': 8,    
        'hidden_dim': 32,
        'dropout': 0.1
    }
    
    config.update(kwargs) 
    
    return RFProjectedAttentionClassifier(
        freq_bins=freq_bins,
        seq_len=seq_len,
        num_classes=num_classes,
        **config
    )

def T_att_base(freq_bins: int, seq_len: int, num_classes: int, **kwargs): # <-- 【改动】include_top 成为显式参数
    """
    创建 'medium' 版本：G=8, att_dim=32，hidden_dim=256。
    """
    config = {
        'groups': 1,
        'att_dim': 16,
        'hidden_dim': 64,
        'dropout': 0.1
    }

    config.update(kwargs)

    return RFProjectedAttentionClassifier(
        freq_bins=freq_bins,
        seq_len=seq_len,
        num_classes=num_classes,
        **config
    )

def T_att_large(freq_bins: int, seq_len: int, num_classes: int, **kwargs): # <-- 【改动】include_top 成为显式参数
    """
    创建 'large' 版本：G=8, att_dim=64，hidden_dim=512。
    """
    config = {
        'groups': 1,
        'att_dim': 32,
        'hidden_dim': 128,
        'dropout': 0.1
    }
    
    config.update(kwargs)

    return RFProjectedAttentionClassifier(
        freq_bins=freq_bins,
        seq_len=seq_len,
        num_classes=num_classes,
        **config
    )

__all__ = [
    'RelativePositionBias',
    'GroupedProjectedAttentionRF',
    'RFProjectedAttentionClassifier',
    'T_att_nano', 'T_att_mini', 'T_att_tiny', 'T_att_small', 'T_att_base', 'T_att_large'
]

if __name__ == "__main__":
    from torchsummary import summary
    from thop import profile
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for model_func, name in [
        (T_att_nano, "T_att_nano"),
        (T_att_mini, "T_att_mini"),
        (T_att_tiny, "T_att_tiny"),
        (T_att_small, "T_att_small"),
        (T_att_base, "T_att_base"),
        (T_att_large, "T_att_large")
    ]:
        print(f"--- Testing {name} ---")
        model = model_func(freq_bins=512, seq_len=512, num_classes=10)
        model = model.to(device)
        inputs = torch.randn(1, 1, 512, 512).to(device) # 1D 频谱向量
        flops, params = profile(model, (inputs,), verbose=False)
        print(name, 'flops: ', flops, 'params: ', params)