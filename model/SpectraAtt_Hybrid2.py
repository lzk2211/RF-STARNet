import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List

# --- (RelativePositionBias 和 PositionalEncoding 保持不变) ---
# ... (此处省略未修改的 RelativePositionBias 和 PositionalEncoding 代码)
class RelativePositionBias(nn.Module):
    def __init__(self, seq_len: int, num_heads: int = 1):
        super().__init__()
        self.seq_len = int(seq_len)
        self.num_heads = int(num_heads)
        self.num_rel_pos = 2 * self.seq_len - 1
        self.relative_bias = nn.Parameter(torch.zeros(self.num_heads, self.num_rel_pos))
        nn.init.trunc_normal_(self.relative_bias, std=0.02)
        coords = torch.arange(self.seq_len)
        rel_pos = coords[None, :] - coords[:, None]
        rel_pos += self.seq_len - 1
        self.register_buffer("rel_pos_index", rel_pos)

    def forward(self, q_len: int, k_len: Optional[int] = None) -> torch.Tensor:
        if k_len is None: k_len = q_len
        q_len, k_len = int(q_len), int(k_len)
        if q_len == k_len == self.seq_len:
            idx = self.rel_pos_index
            return self.relative_bias[:, idx]
        device = self.relative_bias.device
        S = self.seq_len
        denom_q = max(q_len - 1, 1)
        denom_k = max(k_len - 1, 1)
        q_pos_ref = torch.round(torch.linspace(0, S - 1, steps=q_len, device=device))
        k_pos_ref = torch.round(torch.linspace(0, S - 1, steps=k_len, device=device))
        rel = q_pos_ref[:, None] - k_pos_ref[None, :]
        rel = rel.clamp(min=-(S - 1), max=(S - 1)).to(torch.long)
        idx = (rel + (S - 1)).to(torch.long)
        return self.relative_bias[:, idx]

class PositionalEncoding(nn.Module):
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(
            0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)
    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
#  方案一：TSLANet 启发的“多尺度”局部卷积分支
# ---------------------------------------------------------------------------
class MultiKernelTemporalConvBranch(nn.Module):
    """
    受 TSLANet 的 ICB 启发的“多尺度”局部时间卷积分支。
    使用并行的、不同核大小的 1D 深度可分离卷积，
    以自动学习和匹配不同长度的同步序列模式。
    
    输入: (B, T, dim)
    输出: (B, T, dim)
    """
    def __init__(self, dim: int, kernels: List[int] = [3, 7, 11]):
        super().__init__()
        self.kernels = kernels
        self.num_branches = len(kernels)
        
        # 确保维度可以被分支数量整除
        assert dim % self.num_branches == 0, \
            f"Dimension {dim} must be divisible by the number of kernels {self.num_branches}"
        self.split_dim = dim // self.num_branches
        
        # 为每个尺度的核创建一个深度卷积
        self.convs = nn.ModuleList()
        for k in self.kernels:
            self.convs.append(
                nn.Conv1d(
                    in_channels=self.split_dim,
                    out_channels=self.split_dim,
                    kernel_size=k,
                    padding=(k - 1) // 2,
                    groups=self.split_dim, # 深度卷积
                    bias=False
                )
            )
        
        self.pointwise = nn.Linear(dim, dim, bias=False) # 逐点卷积
        self.act = nn.GELU()
        self.layer_norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x 形状: (B, T, dim)
        x_res = x # 残差连接
        x = self.layer_norm(x)
        
        # 沿特征维度分割，为每个卷积分支准备数据
        # x_splits: 元组, 包含 num_branches 个 (B, T, split_dim) 张量
        x_splits = x.split(self.split_dim, dim=-1)
        
        outs = []
        for i, x_s in enumerate(x_splits):
            # 转换为 (B, split_dim, T) 以适应 Conv1d
            x_s_permuted = x_s.permute(0, 2, 1)
            
            # 应用对应的深度卷积
            x_conv = self.convs[i](x_s_permuted)
            
            # 转换回 (B, T, split_dim)
            outs.append(x_conv.permute(0, 2, 1))
        
        # 重新拼接 (B, T, dim)
        x = torch.cat(outs, dim=-1)
        
        x = self.act(x)
        x = self.pointwise(x) # (B, T, dim)
        
        return x + x_res # 添加残差

# ---------------------------------------------------------------------------
#  混合注意力模块 (更新为使用 MultiKernel)
# ---------------------------------------------------------------------------
class HybridGroupedAttentionRF(nn.Module):
    """
    融合了“全局”自注意力和“多尺度局部”卷积的混合注意力模块。
    """
    def __init__(self,
                 freq_bins: int,
                 att_dim: int,
                 groups: int,
                 seq_len: int,
                 k_len: int,
                 use_pos: bool = True,
                 dropout: float = 0.0,
                 local_kernels: List[int] = [3, 7, 11]): # *** 新增 ***
        super().__init__()
        assert freq_bins % groups == 0, "freq_bins must be divisible by groups"
        self.freq_bins = freq_bins
        self.att_dim = att_dim
        self.groups = groups
        self.gdim = freq_bins // groups
        self.seq_len = seq_len
        self.scale = 1.0 / math.sqrt(att_dim)
        
        self.layer_norm = nn.LayerNorm(freq_bins)

        # --- 1. 全局自注意力分支 (不变) ---
        self.qs = nn.ModuleList([nn.Linear(self.gdim, att_dim, bias=False) for _ in range(groups)])
        self.ks = nn.ModuleList([nn.Linear(self.gdim, att_dim, bias=False) for _ in range(groups)])
        self.vs = nn.ModuleList([nn.Linear(self.gdim, self.gdim, bias=False) for _ in range(groups)])
        self.rel_pos_bias = RelativePositionBias(seq_len=seq_len, num_heads=1)
        
        # --- 2. TSLANet 启发的“多尺度”局部分支 (*** 修改 ***) ---
        self.local_branches = nn.ModuleList([
            MultiKernelTemporalConvBranch(dim=self.gdim, kernels=local_kernels) 
            for _ in range(groups)
        ])

        # --- 3. 融合投影层 (不变) ---
        self.os = nn.ModuleList([nn.Linear(self.gdim, self.gdim, bias=False) for _ in range(groups)])
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ... (forward 的前半部分不变) ...
        B, C, F, T = x.shape
        x = x.squeeze(1)  # (B, F, T)
        x_t = x.permute(0, 2, 1)  # (B, T, F)
        x_t = self.layer_norm(x_t)  # (B, T, F)
        B, T2, D = x_t.shape
        G = self.groups
        gdim = self.gdim
        xg = x_t.view(B, T2, G, gdim).permute(0, 2, 1, 3)  # (B, G, T, gdim)

        outs = []
        for g in range(G):
            xg_g = xg[:, g]  # (B, T, gdim)

            # --- 1. 全局分支 (Global Path - Self-Attention) (不变) ---
            q = self.qs[g](xg_g)
            k = self.ks[g](xg_g)
            v = self.vs[g](xg_g)
            att = torch.matmul(q, k.transpose(1, 2)) * self.scale
            bias = self.rel_pos_bias(T2, T2).to(att.device)
            att = att + bias 
            att = torch.softmax(att, dim=-1)
            att = self.dropout(att)
            out_global = torch.matmul(att, v) # (B, T, gdim)

            # --- 2. 局部分支 (Local Path - MultiKernel Conv) (*** 修改 ***) ---
            out_local = self.local_branches[g](xg_g) # (B, T, gdim)
            
            # --- 3. 融合 (Fusion) 与 投影 (Projection) (不变) ---
            out_g = self.os[g](out_global + out_local)
            outs.append(out_g)

        out = torch.cat(outs, dim=-1)  # (B, T, F)
        out = out.permute(0, 2, 1).contiguous()  # (B, F, T)
        return out

# ---------------------------------------------------------------------------
#  分类器封装 (更新以传递 local_kernels)
# ---------------------------------------------------------------------------
class RFProjectedAttentionClassifier(nn.Module):
    def __init__(self,
                 freq_bins: int,
                 seq_len: int,
                 num_classes: int = 25,
                 groups: int = 8,
                 att_dim: int = 32,
                 k_len: int = 64,
                 hidden_dim: Optional[int] = None,
                 use_pos: bool = True,
                 dropout: float = 0.1,
                 local_kernels: List[int] = [3, 7, 11]): # *** 新增 ***
        super().__init__()
        
        self.att = HybridGroupedAttentionRF(
            freq_bins=freq_bins,
            att_dim=att_dim,
            groups=groups,
            seq_len=seq_len,
            k_len=k_len,
            use_pos=use_pos,
            dropout=dropout,
            local_kernels=local_kernels # *** 传递 ***
        )
        
        if hidden_dim is None: hidden_dim = max(128, freq_bins // 2)
        self.fc = nn.Sequential(
            nn.Linear(freq_bins, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.att(x)  # (B, F, T)
        pooled = out.mean(dim=-1)  # (B, F)
        logits = self.fc(pooled)   # (B, num_classes)
        return logits

# ---------------------------------------------------------------------------
#  模型工厂函数 (更新)
# ---------------------------------------------------------------------------
def rf_projected_attention_large(freq_bins: int, seq_len: int, num_classes: int, **kwargs):
    config = {
        'groups': 8,
        'att_dim': 64,
        'k_len': 128,
        'hidden_dim': 512,
        'dropout': 0.1,
        'local_kernels': [3, 7, 11, 15] # 'Large' 模型可以使用更多或更大的核
    }
    config.update(kwargs)
    return RFProjectedAttentionClassifier(
        freq_bins=freq_bins, seq_len=seq_len, num_classes=num_classes, **config
    )

def rf_projected_attention_medium(freq_bins: int, seq_len: int, num_classes: int, **kwargs):
    config = {
        'groups': 8,
        'att_dim': 32,
        'k_len': 64,
        'hidden_dim': 256,
        'dropout': 0.1,
        'local_kernels': [3, 7]
    }
    config.update(kwargs)
    return RFProjectedAttentionClassifier(
        freq_bins=freq_bins, seq_len=seq_len, num_classes=num_classes, **config
    )

# ... (其他工厂函数同理) ...

# ---------------------------------------------------------------------------
#  测试代码
# ---------------------------------------------------------------------------
if False:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # model = rf_projected_attention_medium(
    #     freq_bins=512, 
    #     seq_len=512, 
    #     num_classes=25,
    #     local_kernels=[3, 7] # 演示覆盖
    # ).to(device)

    model = rf_projected_attention_large(
        freq_bins=512, 
        seq_len=512, 
        num_classes=25
    ).to(device)


    x = torch.randn(16, 1, 512, 512).to(device)  # (B, C, F, T)

    # 尝试计算参数量
    try:
        from thop import profile
        inputs = torch.randn(1, 1, 512, 512).to(device)
        flops, params = profile(model, (inputs,))
        print(f"Model: large (hybrid)")
        print(f"Params: {params/1e6:.2f} M")
        print(f"FLOPs: {flops/1e9:.2f} G")
    except ImportError:
        print("\n[Warning] thop 未安装，跳过 FLOPs/Params 计算。")
        print("          请运行: pip install thop")