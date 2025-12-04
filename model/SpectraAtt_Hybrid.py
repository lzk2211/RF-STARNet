import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# ---------------------------------------------------------------------------
#  未修改的辅助模块 (来自您的原始代码)
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
#  TSLANet 启发的“局部”卷积分支
# ---------------------------------------------------------------------------
class TemporalConvBranch(nn.Module):
    """
    受 TSLANet 的 ICB 启发的局部时间卷积分支。
    使用 1D 深度可分离卷积来捕捉局部时间模式 (例如 ZC 序列特性)。
    
    输入: (B, T, dim)
    输出: (B, T, dim)
    """
    def __init__(self, dim: int, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            groups=dim,  # 深度卷积 (Depthwise)
            bias=False
        )
        self.pointwise = nn.Linear(dim, dim, bias=False) # 逐点卷积 (Pointwise)
        self.act = nn.GELU()
        self.layer_norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x 形状: (B, T, dim)
        x_res = x # 残差连接
        
        x = self.layer_norm(x)
        
        # 转换为 (B, dim, T) 以适应 Conv1d
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        
        # 转换回 (B, T, dim)
        x = x.permute(0, 2, 1)
        
        x = self.act(x)
        x = self.pointwise(x)
        
        return x + x_res # 添加残差

# ---------------------------------------------------------------------------
#  融合了 TSLANet 思想的“混合”注意力模块
# ---------------------------------------------------------------------------
class HybridGroupedAttentionRF(nn.Module):
    """
    融合了“全局”自注意力和“局部”卷积的混合注意力模块。
    
    - 全局分支 (Global): 来自您原有的 T-T 自注意力机制。
    - 局部分支 (Local): 受 TSLANet (ICB) 启发的 1D 时间卷积分支。
    
    专为输入 (B, C, F, T) 设计。
    """

    def __init__(self,
                 freq_bins: int,
                 att_dim: int,
                 groups: int,
                 seq_len: int,
                 k_len: int,
                 use_pos: bool = True,
                 dropout: float = 0.0,
                 local_kernel_size: int = 7): # 新增：局部卷积核大小
        super().__init__()
        assert freq_bins % groups == 0, "freq_bins must be divisible by groups"
        self.freq_bins = freq_bins
        self.att_dim = att_dim
        self.groups = groups
        self.gdim = freq_bins // groups
        self.seq_len = seq_len
        self.k_len = k_len # 注意：在纯自注意力中 k_len 未被使用
        self.scale = 1.0 / math.sqrt(att_dim)
        self.use_pos = use_pos # 注意：use_pos 未在 forward 中使用

        self.pos_encoding = PositionalEncoding(freq_bins, dropout) # 注意：pos_encoding 未在 forward 中使用
        self.layer_norm = nn.LayerNorm(freq_bins)

        # --- 1. 全局自注意力分支 (来自您的原始代码) ---
        self.qs = nn.ModuleList([nn.Linear(self.gdim, att_dim, bias=False) for _ in range(groups)])
        self.ks = nn.ModuleList([nn.Linear(self.gdim, att_dim, bias=False) for _ in range(groups)])
        self.vs = nn.ModuleList([nn.Linear(self.gdim, self.gdim, bias=False) for _ in range(groups)])
        self.rel_pos_bias = RelativePositionBias(seq_len=seq_len, num_heads=1)
        
        # --- 2. TSLANet 启发的局部分支 ---
        self.local_branches = nn.ModuleList([
            TemporalConvBranch(dim=self.gdim, kernel_size=local_kernel_size) 
            for _ in range(groups)
        ])

        # --- 3. 融合投影层 (使用您原有的 os) ---
        self.os = nn.ModuleList([nn.Linear(self.gdim, self.gdim, bias=False) for _ in range(groups)])

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # 删除了 Pk, Pv, time_pos 和 _resample_P，因为它们在您的新版代码中未被使用
        # (如果您需要它们，可以加回来)

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
        for g in range(G):
            xg_g = xg[:, g]  # (B, T, gdim)

            # --- 1. 全局分支 (Global Path - Self-Attention) ---
            q = self.qs[g](xg_g)  # (B, T, att_dim)
            k = self.ks[g](xg_g)  # (B, T, att_dim)
            v = self.vs[g](xg_g)  # (B, T, gdim)

            att = torch.matmul(q, k.transpose(1, 2)) * self.scale

            bias = self.rel_pos_bias(T2, T2).to(att.device)  # (1, T2, T2)
            att = att + bias 

            att = torch.softmax(att, dim=-1)
            att = self.dropout(att)

            out_global = torch.matmul(att, v) # (B, T, gdim)

            # --- 2. 局部分支 (Local Path - TSLANet-inspired Conv) ---
            out_local = self.local_branches[g](xg_g) # (B, T, gdim)
            
            # --- 3. 融合 (Fusion) 与 投影 (Projection) ---
            # 简单地将全局和局部特征相加，然后通过输出投影
            out_g = self.os[g](out_global + out_local)  # (B, T, gdim)
            outs.append(out_g)

        out = torch.cat(outs, dim=-1)  # (B, T, F)
        out = out.permute(0, 2, 1).contiguous()  # (B, F, T)
        return out

# ---------------------------------------------------------------------------
#  分类器封装 (更新以使用 Hybrid 模块)
# ---------------------------------------------------------------------------
class RFProjectedAttentionClassifier(nn.Module):
    """
    将 HybridGroupedAttentionRF 封装为一个分类器：
    输入 x: (B, C, F, T)
    输出 logits: (B, num_classes)
    """
    def __init__(self,
                 freq_bins: int,
                 seq_len: int,
                 num_classes: int = 25,
                 groups: int = 8,
                 att_dim: int = 32,
                 k_len: int = 64,  # k_len 在 Hybrid 模块中未使用，但保留以兼容工厂函数
                 hidden_dim: Optional[int] = None,
                 use_pos: bool = True, # use_pos 在 Hybrid 模块中未使用，但保留
                 dropout: float = 0.1,
                 local_kernel_size: int = 7): # 新增：传递给 Hybrid 模块
        super().__init__()
        
        # *** 修改点：使用新的 Hybrid 模块 ***
        self.att = HybridGroupedAttentionRF(
            freq_bins=freq_bins,
            att_dim=att_dim,
            groups=groups,
            seq_len=seq_len,
            k_len=k_len,
            use_pos=use_pos,
            dropout=dropout,
            local_kernel_size=local_kernel_size # 传递卷积核大小
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

# ---------------------------------------------------------------------------
#  模型工厂函数 (ResNet 风格) - 保持不变
# ---------------------------------------------------------------------------
# 这些工厂函数现在会自动创建使用 Hybrid 模块的分类器
# 您可以通过 kwargs 传递 local_kernel_size 来覆盖默认值 (e.g., local_kernel_size=11)

def rf_projected_attention_small(freq_bins: int, seq_len: int, num_classes: int, **kwargs):
    config = {
        'groups': 8,      
        'att_dim': 16,    
        'k_len': 32,      
        'hidden_dim': 128,
        'dropout': 0.1,
        'local_kernel_size': 7 # 默认局部核大小
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
        'local_kernel_size': 7 # 默认局部核大小
    }
    config.update(kwargs)
    return RFProjectedAttentionClassifier(
        freq_bins=freq_bins, seq_len=seq_len, num_classes=num_classes, **config
    )

def rf_projected_attention_large(freq_bins: int, seq_len: int, num_classes: int, **kwargs):
    config = {
        'groups': 8,
        'att_dim': 64,
        'k_len': 128,
        'hidden_dim': 512,
        'dropout': 0.1,
        'local_kernel_size': 7 # 默认局部核大小
    }
    config.update(kwargs)
    return RFProjectedAttentionClassifier(
        freq_bins=freq_bins, seq_len=seq_len, num_classes=num_classes, **config
    )

# ---------------------------------------------------------------------------
#  测试代码
# ---------------------------------------------------------------------------
if False:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 使用 "large" 工厂函数创建混合模型
    model = rf_projected_attention_medium(
        freq_bins=512, 
        seq_len=512, 
        num_classes=25,
        local_kernel_size=11 # 演示覆盖默认核大小
    ).to(device)

    print(model) # 打印模型结构，检查 HybridGroupedAttentionRF 和 TemporalConvBranch

    x = torch.randn(16, 1, 512, 512).to(device)  # (B, C, F, T)
    out = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output logits shape: {out.shape}") # 应该为 (16, 25)

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