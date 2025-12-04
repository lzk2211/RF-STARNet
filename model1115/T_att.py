import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

class RelativePositionBias(nn.Module):
    """
    相对位置偏置（Relative Position Bias）。
    ... (此模块无改动) ...
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
    分组注意力（无投影版）。
    
    【重构】：此模块现在只负责注意力计算。
    输入形状 (B, C, F, T_new)，即 *已经下采样* 后的张量。
    输出形状 (B, F, T_new)。
    """

    def __init__(self,
                 freq_bins: int,
                 att_dim: int,
                 groups: int,
                 seq_len: int, # <-- 【改动】这里的 seq_len 是 T_new (例如 128 或 32)
                 use_pos: bool = True,
                 dropout: float = 0.0):
        super().__init__()
        assert freq_bins % groups == 0, "freq_bins must be divisible by groups"
        self.freq_bins = freq_bins
        self.att_dim = att_dim
        self.groups = groups
        self.gdim = freq_bins // groups
        self.seq_len = seq_len # <-- 这是 T_new
        self.scale = 1.0 / math.sqrt(att_dim)
        self.use_pos = use_pos

        # per-group linear projections
        self.qs = nn.ModuleList([nn.Linear(self.gdim, att_dim, bias=False) for _ in range(groups)])
        self.ks = nn.ModuleList([nn.Linear(self.gdim, att_dim, bias=False) for _ in range(groups)])
        self.vs = nn.ModuleList([nn.Linear(self.gdim, self.gdim, bias=False) for _ in range(groups)])
        self.os = nn.ModuleList([nn.Linear(self.gdim, self.gdim, bias=False) for _ in range(groups)])

        # 【改动】移除 stem_pool
        
        # 相对位置偏置：按组（num_heads = groups）各自一份
        # seq_len (T_new) 直接在此处使用
        self.rel_pos_bias = RelativePositionBias(seq_len=seq_len, num_heads=groups)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.layer_norm = nn.LayerNorm(freq_bins)
        # 位置编码模块暂不启用，如需可外部接入

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, F, T_new) <-- 【改动】输入已是下采样后的 x_stem
        returns: out (B, F, T_new)
        """
        # 【改动】移除 x_stem = self.stem_pool(x)
        
        B, C, F, T = x.shape # T 已经是 T_new
        x = x.squeeze(1)  # (B, F, T)
        assert F == self.freq_bins, f"Input freq dim {F} must equal freq_bins {self.freq_bins}"
        # 增加一个检查，确保传入的 T 与模块配置的 seq_len (T_new) 一致
        assert T == self.seq_len, f"Input T {T} mismatch with configured seq_len {self.seq_len}"

        x_t = x
        x_t = x_t.permute(0, 2, 1)  # (B, T, F)
        x_t = self.layer_norm(x_t)  # normalize over F

        B, T2, D = x_t.shape  # T2 == T, D == F
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
    分类器封装：
    1. Stem: 使用 AvgPool2d 在时间 T 维度上进行下采样。
    2. Attention: 使用 GroupedProjectedAttentionRF 获得 (B, F, T_new)
    3. Head: (可选) 时间维度平均池化 (B, F)，接 MLP 输出 logits。
    """

    def __init__(self,
                 freq_bins: int,
                 seq_len: int,          # <-- 原始 T (例如 512)
                 num_classes: int = 25,
                 groups: int = 8,
                 att_dim: int = 32,
                 hidden_dim: Optional[int] = None,
                 use_pos: bool = True,
                 dropout: float = 0.1,
                 stem_pool_factor: int = 8,   # <-- 【改动】Stem 参数移回此处
                 include_top: bool = True): # <-- 【新增】include_top 参数
        super().__init__()
        
        self.include_top = include_top
        
        # 1. Stem: 【改动】移回此处
        self.stem_pool = nn.AvgPool2d(
            kernel_size=(1, stem_pool_factor), 
            stride=(1, stem_pool_factor)
        )
        
        # 计算池化后的新序列长度
        assert seq_len % stem_pool_factor == 0, "seq_len must be divisible by stem_pool_factor"
        new_seq_len = seq_len // stem_pool_factor
        
        # 2. Attention: 
        # 【改动】att 模块传入 new_seq_len
        self.att = GroupedProjectedAttentionRF(
            freq_bins=freq_bins,
            att_dim=att_dim,
            groups=groups,
            seq_len=new_seq_len, # <-- 传入新的 T (例如 128)
            use_pos=use_pos,
            dropout=dropout
        )
        
        if hidden_dim is None:
            hidden_dim = max(128, freq_bins // 2)

        # 3. Head (分类头)
        # 【新增】根据 include_top 决定是否创建 fc
        if self.include_top:
            self.fc = nn.Sequential(
                nn.Linear(freq_bins, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes)
            )
        else:
            # 如果不需要 top，可以设为 Identity 或 None
            self.fc = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, F, T)  (例如 (B, 1, 512, 512))
        returns logits or features: 
            - (B, num_classes) if include_top=True
            - (B, F, T_new)    if include_top=False
        """
        # 1. 先通过 Stem 下采样
        # (B, 1, 512, 512) -> (B, 1, 512, 128)
        x_stem = self.stem_pool(x)
        
        # 2. 再送入注意力
        # (B, 1, 512, 128) -> (B, 512, 128)
        out = self.att(x_stem)
        
        # 【新增】根据 include_top 决定是否返回
        if not self.include_top:
            return out  # <-- 返回 (B, F, T_new)
        
        # 3. 池化和分类
        pooled = out.mean(dim=-1)  # (B, 512)
        logits = self.fc(pooled)   # (B, num_classes)
        return logits

# =============================================================================
#  模型工厂函数 (ResNet 风格)
# =============================================================================

def rf_att_mini(freq_bins: int, seq_len: int, num_classes: int, 
                include_top: bool = True, **kwargs): # <-- 【改动】include_top 成为显式参数

    config = {
        'groups': 128,
        'att_dim': 4,
        'hidden_dim': 8,
        'dropout': 0.1,
        'stem_pool_factor': 32
    }
    
    # 允许 kwargs 覆盖默认配置 (例如 dropout=0.0)
    config.update(kwargs) 
    
    return RFProjectedAttentionClassifier(
        freq_bins=freq_bins,
        seq_len=seq_len,
        num_classes=num_classes,
        # stem_pool_factor=stem_pool_factor,
        include_top=include_top, # <-- 直接传入
        **config
    )

def rf_att_tiny(freq_bins: int, seq_len: int, num_classes: int, 
                include_top: bool = True, **kwargs): # <-- 【改动】include_top 成为显式参数
    """
    创建 'tiny' 版本：G=64, att_dim=4, hidden_dim=16。
    """
    config = {
        'groups': 64,
        'att_dim': 4,
        'hidden_dim': 16,
        'dropout': 0.1,
        'stem_pool_factor': 16
    }
    
    # 允许 kwargs 覆盖默认配置 (例如 dropout=0.0)
    config.update(kwargs) 
    
    return RFProjectedAttentionClassifier(
        freq_bins=freq_bins,
        seq_len=seq_len,
        num_classes=num_classes,
        # stem_pool_factor=stem_pool_factor,
        include_top=include_top, # <-- 直接传入
        **config
    )

def rf_att_small(freq_bins: int, seq_len: int, num_classes: int, 
                 include_top: bool = True, **kwargs): # <-- 【改动】include_top 成为显式参数
    """
    创建 'small' 版本：G=8, att_dim=16，hidden_dim=128。
    """
    config = {
        'groups': 8,      
        'att_dim': 16,    
        'hidden_dim': 128,
        'dropout': 0.1,
        'stem_pool_factor': 8
    }
    
    config.update(kwargs) 
    
    return RFProjectedAttentionClassifier(
        freq_bins=freq_bins,
        seq_len=seq_len,
        num_classes=num_classes,
        # stem_pool_factor=stem_pool_factor,
        include_top=include_top, # <-- 直接传入
        **config
    )

def rf_att_medium(freq_bins: int, seq_len: int, num_classes: int, 
                  include_top: bool = True, **kwargs): # <-- 【改动】include_top 成为显式参数
    """
    创建 'medium' 版本：G=8, att_dim=32，hidden_dim=256。
    """
    config = {
        'groups': 8,
        'att_dim': 32,
        'hidden_dim': 256,
        'dropout': 0.1,
        'stem_pool_factor': 8
    }

    config.update(kwargs)

    return RFProjectedAttentionClassifier(
        freq_bins=freq_bins,
        seq_len=seq_len,
        num_classes=num_classes,
        include_top=include_top, # <-- 直接传入
        **config
    )

def rf_att_large(freq_bins: int, seq_len: int, num_classes: int,
                 include_top: bool = True, **kwargs): # <-- 【改动】include_top 成为显式参数
    """
    创建 'large' 版本：G=8, att_dim=64，hidden_dim=512。
    """
    config = {
        'groups': 8,
        'att_dim': 64,
        'hidden_dim': 512,
        'dropout': 0.1,
        'stem_pool_factor': 8
    }
    
    config.update(kwargs)

    return RFProjectedAttentionClassifier(
        freq_bins=freq_bins,
        seq_len=seq_len,
        num_classes=num_classes,
        include_top=include_top, # <-- 直接传入
        **config
    )

__all__ = [
    'RelativePositionBias',
    'GroupedProjectedAttentionRF',
    'RFProjectedAttentionClassifier',
    'rf_att_mini', 'rf_att_tiny', 'rf_att_small', 'rf_att_medium', 'rf_att_large'
]

if __name__ == "__main__":
    from torchsummary import summary
    from thop import profile
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for model_func, name in [
        (rf_att_mini, "rf_att_mini"),
        (rf_att_tiny, "rf_att_tiny"),
        (rf_att_small, "rf_att_small"),
        (rf_att_medium, "rf_att_medium"),
        (rf_att_large, "rf_att_large")
    ]:
        print(f"--- Testing {name} ---")
        model = model_func(freq_bins=512, seq_len=512, num_classes=10)
        model = model.to(device)
        inputs = torch.randn(1, 1, 512, 512).to(device) # 1D 频谱向量
        flops, params = profile(model, (inputs,), verbose=False)
        print(name, 'flops: ', flops, 'params: ', params)