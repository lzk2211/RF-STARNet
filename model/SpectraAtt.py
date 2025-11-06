import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# try:
#     from thop import profile
#     import humanize
#     THOP_AVAILABLE = True
# except ImportError:
#     print("[Warning] thop or humanize not installed. Skipping FLOPs/Params calculation.")
#     print("          Please install with: pip install thop humanize")
#     THOP_AVAILABLE = False

class RelativePositionBias(nn.Module):
    """
    计算相对位置偏置，支持非方阵 (q_len, k_len) 的情形。
    - 参数表按 reference `seq_len` 构建，长度为 2*seq_len-1。
    - 对于任意 (q_len, k_len)，将 q/k 位置映射到参考序列范围 [0, seq_len-1]，
      以此得到整数相对位移并索引到偏置表，返回形状 (num_heads, q_len, k_len)。
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
        返回形状 (num_heads, q_len, k_len) 的偏置矩阵。
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
        # 避免除零
        denom_q = max(q_len - 1, 1)
        denom_k = max(k_len - 1, 1)
        q_pos_ref = torch.round(torch.linspace(0, S - 1, steps=q_len, device=device))  # (q_len,)
        k_pos_ref = torch.round(torch.linspace(0, S - 1, steps=k_len, device=device))  # (k_len,)

        # 计算整数相对位移并映射到 [0, 2S-2]
        rel = q_pos_ref[:, None] - k_pos_ref[None, :]  # (q_len, k_len)
        rel = rel.clamp(min=-(S - 1), max=(S - 1)).to(torch.long)
        idx = (rel + (S - 1)).to(torch.long)  # (q_len, k_len)

        # 索引查表 -> (H, q_len, k_len)
        return self.relative_bias[:, idx]
    
class PositionalEncoding(nn.Module):
    """Positional encoding."""
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        # Create a long enough `P`
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(
            0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)
    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)
    
class GroupedProjectedAttentionRF(nn.Module):
    """
    改写的 GroupedProjectedAttention，专为输入 (B, F, T) 设计。
    """

    def __init__(self,
                 freq_bins: int,
                 att_dim: int,
                 groups: int,
                 seq_len: int,
                 k_len: int,
                 use_pos: bool = True,
                 dropout: float = 0.0):
        super().__init__()
        assert freq_bins % groups == 0, "freq_bins must be divisible by groups"
        self.freq_bins = freq_bins
        self.att_dim = att_dim
        self.groups = groups
        self.gdim = freq_bins // groups
        self.seq_len = seq_len
        self.k_len = k_len
        self.scale = 1.0 / math.sqrt(att_dim)
        self.use_pos = use_pos

        self.pos_encoding = PositionalEncoding(freq_bins, dropout)

        # per-group linear projections
        self.qs = nn.ModuleList([nn.Linear(self.gdim, att_dim, bias=False) for _ in range(groups)])
        self.ks = nn.ModuleList([nn.Linear(self.gdim, att_dim, bias=False) for _ in range(groups)])
        self.vs = nn.ModuleList([nn.Linear(self.gdim, self.gdim, bias=False) for _ in range(groups)])
        self.os = nn.ModuleList([nn.Linear(self.gdim, self.gdim, bias=False) for _ in range(groups)])

        # Pk, Pv: (seq_len, k_len) per group
        self.Pk = nn.ParameterList([nn.Parameter(torch.randn(seq_len, k_len) * (1.0 / math.sqrt(seq_len)))
                                     for _ in range(groups)])
        self.Pv = nn.ParameterList([nn.Parameter(torch.randn(seq_len, k_len) * (1.0 / math.sqrt(seq_len)))
                                     for _ in range(groups)])

        if use_pos:
            self.time_pos = nn.Parameter(torch.zeros(1, seq_len, 1))
        else:
            self.register_parameter('time_pos', None)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.layer_norm = nn.LayerNorm(freq_bins)
        self.encoding_to_sa = nn.Sequential(nn.Linear(128, 128), nn.ReLU())

        self.rel_pos_bias = RelativePositionBias(seq_len=seq_len, num_heads=1)
        
    def _resample_P(self, P: torch.Tensor, target_T: int) -> torch.Tensor:
        """
        P: (seq_len, k_len)
        return: (target_T, k_len) by linear interpolation along seq_len dim if needed.
        """
        if target_T == self.seq_len:
            return P
        Pt = P.t().unsqueeze(0)  # (1, k_len, seq_len)
        Pt_resized = F.interpolate(Pt, size=target_T, mode='linear', align_corners=False)  # (1, k_len, target_T)
        Pt_resized = Pt_resized.squeeze(0)  # (k_len, target_T)
        return Pt_resized.t()  # -> (target_T, k_len)

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

        B, T2, D = x_t.shape  # 现在 T2 == 原始 T, D == F
        G = self.groups
        gdim = self.gdim

        xg = x_t.view(B, T2, G, gdim).permute(0, 2, 1, 3)  # (B, G, T, gdim)

        outs = []
        for g in range(G):
            xg_g = xg[:, g]  # (B, T, gdim)
            q = self.qs[g](xg_g)  # (B, T, att_dim)
            k = self.ks[g](xg_g)  # (B, T, att_dim)
            v = self.vs[g](xg_g)  # (B, T, gdim)

            Pk_g = self._resample_P(self.Pk[g], T2)  # (T2, k_len)
            Pv_g = self._resample_P(self.Pv[g], T2)  # (T2, k_len)

            k_red = torch.einsum('bta,tk->bka', k, Pk_g)  # (B, k_len, att_dim)
            v_red = torch.einsum('btg,tk->bkg', v, Pv_g)  # (B, k_len, gdim)

            att = torch.matmul(q, k_red.transpose(1, 2)) * self.scale

            bias = self.rel_pos_bias(T2, self.k_len).to(att.device)  # (1, T2, k_len)
            att = att + bias 

            att = torch.softmax(att, dim=-1)
            att = self.dropout(att)

            out_g = torch.matmul(att, v_red)
            out_g = self.os[g](out_g)  # (B, T, gdim)
            outs.append(out_g)

        out = torch.cat(outs, dim=-1)  # (B, T, F)
        out = out.permute(0, 2, 1).contiguous()  # (B, F, T)
        return out


class RFProjectedAttentionClassifier(nn.Module):
    """
    将 GroupedProjectedAttentionRF 封装为一个分类器：
    输入 x: (B, C, F, T)
    输出 logits: (B, num_classes)
    """

    def __init__(self,
                 freq_bins: int,
                 seq_len: int,
                 num_classes: int = 25,
                 groups: int = 8,
                 att_dim: int = 32,
                 k_len: int = 64,
                 hidden_dim: Optional[int] = None,
                 use_pos: bool = True,
                 dropout: float = 0.1):
        super().__init__()
        self.att = GroupedProjectedAttentionRF(
            freq_bins=freq_bins,
            att_dim=att_dim,
            groups=groups,
            seq_len=seq_len,
            k_len=k_len,
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


# =============================================================================
#  模型工厂函数 (ResNet 风格)
# =============================================================================

def rf_projected_attention_small(freq_bins: int, seq_len: int, num_classes: int, **kwargs):
    """
    创建 'small' 版本的 RFProjectedAttentionClassifier。
    
    - att_dim: 16
    - k_len: 32
    - hidden_dim: 128
    """
    config = {
        'groups': 8,      
        'att_dim': 16,    
        'k_len': 32,      
        'hidden_dim': 128,
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

def rf_projected_attention_medium(freq_bins: int, seq_len: int, num_classes: int, **kwargs):
    """
    创建 'medium' 版本的 RFProjectedAttentionClassifier。
    
    - att_dim: 32
    - k_len: 64
    - hidden_dim: 256
    """
    config = {
        'groups': 8,
        'att_dim': 32,
        'k_len': 64,
        'hidden_dim': 256,
        'dropout': 0.1
    }
    config.update(kwargs)

    return RFProjectedAttentionClassifier(
        freq_bins=freq_bins,
        seq_len=seq_len,
        num_classes=num_classes,
        **config
    )

def rf_projected_attention_large(freq_bins: int, seq_len: int, num_classes: int, **kwargs):
    """
    创建 'large' 版本的 RFProjectedAttentionClassifier。
    
    - att_dim: 64
    - k_len: 128
    - hidden_dim: 512
    """
    config = {
        'groups': 8,
        'att_dim': 64,
        'k_len': 128,
        'hidden_dim': 512,
        'dropout': 0.1
    }
    config.update(kwargs)

    return RFProjectedAttentionClassifier(
        freq_bins=freq_bins,
        seq_len=seq_len,
        num_classes=num_classes,
        **config
    )

if False:
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    model = rf_projected_attention_large(freq_bins=512, seq_len=128, num_classes=25).to(device)

    x = torch.randn(16, 1, 512, 512).to(device)  # batch_size=16
    out = model(x)
    print(out.shape)

    from torchsummary import summary
    from thop import profile

    # summary(model, input_size=(1, 512))
    inputs = torch.randn(1, 1, 512, 512).to(device)
    flops, params = profile(model, (inputs,))
    print('flops: ', flops, 'params: ', params)
