import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

class RelativePositionBias(nn.Module):
    """
    Relative Position Bias.
    ... (This module has no changes) ...
    """
    def __init__(self, seq_len: int, num_heads: int = 1):
        super().__init__()
        self.seq_len = int(seq_len)
        self.num_heads = int(num_heads)

        # Support distance range [-S+1, S-1]
        self.num_rel_pos = 2 * self.seq_len - 1

        # Each relative distance corresponds to a bias value
        self.relative_bias = nn.Parameter(torch.zeros(self.num_heads, self.num_rel_pos))
        nn.init.trunc_normal_(self.relative_bias, std=0.02)

        # Precompute index table for square case (S, S), construct dynamically for non-square matrix
        coords = torch.arange(self.seq_len)
        rel_pos = coords[None, :] - coords[:, None]  # i - j
        rel_pos += self.seq_len - 1  # shift to [0, 2S-2]
        self.register_buffer("rel_pos_index", rel_pos)

    def forward(self, q_len: int, k_len: Optional[int] = None) -> torch.Tensor:
        """
        Return bias matrix, shape (num_heads, q_len, k_len).
        If k_len is omitted or equal to q_len, degenerate to square matrix (q_len, q_len).
        """
        if k_len is None:
            k_len = q_len

        q_len = int(q_len)
        k_len = int(k_len)

        # Fast path: Consistent with reference length and square matrix
        if q_len == k_len == self.seq_len:
            idx = self.rel_pos_index  # (S, S)
            return self.relative_bias[:, idx]  # (H, S, S)

        device = self.relative_bias.device

        # Linear map query/key position to reference sequence [0, S-1]
        S = self.seq_len
        q_pos_ref = torch.round(torch.linspace(0, S - 1, steps=q_len, device=device))  # (q_len,)
        k_pos_ref = torch.round(torch.linspace(0, S - 1, steps=k_len, device=device))  # (k_len,)

        # Calculate integer relative displacement and map to [0, 2S-2]
        rel = q_pos_ref[:, None] - k_pos_ref[None, :]  # (q_len, k_len)
        rel = rel.clamp(min=-(S - 1), max=(S - 1)).to(torch.long)
        idx = (rel + (S - 1)).to(torch.long)  # (q_len, k_len)

        # Look up table index -> (H, q_len, k_len)
        return self.relative_bias[:, idx]
    
class GroupedProjectedAttentionRF(nn.Module):
    """
    Grouped Attention (No projection version).
    
    [Refactor]: This module is now only responsible for attention calculation.
    Input shape (B, C, F, T_new), i.e., tensor *already downsampled*.
    Output shape (B, F, T_new).
    """

    def __init__(self,
                 freq_bins: int,
                 att_dim: int,
                 groups: int,
                 seq_len: int, # <-- [Change] seq_len here is T_new (e.g., 128 or 32)
                 use_pos: bool = True,
                 dropout: float = 0.0):
        super().__init__()
        assert freq_bins % groups == 0, "freq_bins must be divisible by groups"
        self.freq_bins = freq_bins
        self.att_dim = att_dim
        self.groups = groups
        self.gdim = freq_bins // groups
        self.seq_len = seq_len # <-- This is T_new
        self.scale = 1.0 / math.sqrt(att_dim)
        self.use_pos = use_pos

        # per-group linear projections
        self.qs = nn.ModuleList([nn.Linear(self.gdim, att_dim, bias=False) for _ in range(groups)])
        self.ks = nn.ModuleList([nn.Linear(self.gdim, att_dim, bias=False) for _ in range(groups)])
        self.vs = nn.ModuleList([nn.Linear(self.gdim, self.gdim, bias=False) for _ in range(groups)])
        self.os = nn.ModuleList([nn.Linear(self.gdim, self.gdim, bias=False) for _ in range(groups)])

        # [Change] Remove stem_pool
        
        # Relative Position Bias: one copy per group (num_heads = groups)
        # Use seq_len (T_new) directly here
        self.rel_pos_bias = RelativePositionBias(seq_len=seq_len, num_heads=groups)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.layer_norm = nn.LayerNorm(freq_bins)
        # Positional encoding module is not enabled for now, can be accessed externally if needed

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, F, T_new) <-- [Change] Input is already downsampled x_stem
        returns: out (B, F, T_new)
        """
        # [Change] Remove x_stem = self.stem_pool(x)
        
        B, C, F, T = x.shape # T is already T_new
        x = x.squeeze(1)  # (B, F, T)
        assert F == self.freq_bins, f"Input freq dim {F} must equal freq_bins {self.freq_bins}"
        # Add a check to ensure input T matches module configured seq_len (T_new)
        assert T == self.seq_len, f"Input T {T} mismatch with configured seq_len {self.seq_len}"

        x_t = x
        x_t = x_t.permute(0, 2, 1)  # (B, T, F)
        x_t = self.layer_norm(x_t)  # normalize over F

        B, T2, D = x_t.shape  # T2 == T, D == F
        G = self.groups
        gdim = self.gdim

        xg = x_t.view(B, T2, G, gdim).permute(0, 2, 1, 3)  # (B, G, T, gdim)

        outs = []
        # Precompute relative position bias for all groups to reduce redundant construction
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
    # config = {
    #     'groups': 64,
    #     'att_dim': 2,
    #     'hidden_dim': 4,
    #     'dropout': 0.1,
    #     'stem_pool_factor': 64
    # }
def T_att_nano(freq_bins: int, seq_len: int, num_classes: int, 
                include_top: bool = True, **kwargs): # <-- 【改动】include_top 成为显式参数

    config = {
        'groups': 128,
        'att_dim': 2,
        'hidden_dim': 4,
        'dropout': 0.1,
        'stem_pool_factor': 64
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


def T_att_mini(freq_bins: int, seq_len: int, num_classes: int, 
                include_top: bool = True, **kwargs): # <-- 【改动】include_top 成为显式参数

    config = {
        'groups': 64,
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

def T_att_tiny(freq_bins: int, seq_len: int, num_classes: int, 
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

def T_att_small(freq_bins: int, seq_len: int, num_classes: int, 
                 include_top: bool = True, **kwargs): # <-- 【改动】include_top 成为显式参数
    """
    创建 'small' 版本：G=8, att_dim=16，hidden_dim=128。
    """
    config = {
        'groups': 32,      
        'att_dim': 8,    
        'hidden_dim': 32,
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

def T_att_base(freq_bins: int, seq_len: int, num_classes: int, 
                  include_top: bool = True, **kwargs): # <-- 【改动】include_top 成为显式参数
    """
    创建 'medium' 版本：G=8, att_dim=32，hidden_dim=256。
    """
    config = {
        'groups': 16,
        'att_dim': 16,
        'hidden_dim': 64,
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

def T_att_large(freq_bins: int, seq_len: int, num_classes: int,
                 include_top: bool = True, **kwargs): # <-- 【改动】include_top 成为显式参数
    """
    创建 'large' 版本：G=8, att_dim=64，hidden_dim=512。
    """
    config = {
        'groups': 8,
        'att_dim': 32,
        'hidden_dim': 128,
        'dropout': 0.1,
        'stem_pool_factor': 16
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

    model = T_att_small(freq_bins=512, seq_len=512, num_classes=10).to(device)
    dummy_input = torch.randn(1, 1, 512, 512).to(device)

    # 4. 导出为 ONNX
    torch.onnx.export(model,               # 你要导出的模型
                    dummy_input,         # 一个示例输入
                    "model.onnx",        # 导出的文件名
                    input_names=['input'], # 可选：指定输入名
                    output_names=['output']) # 可选：指定输出名

    print("模型已成功导出为 model.onnx")
