import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# def position_coding(x: torch.Tensor) -> torch.Tensor:
#     """
#     生成与输入 x 同步的正弦-余弦位置编码。
#     期望 x 的最后两维分别是 (num_token, num_dims)，返回形状 (1, num_token, num_dims)。
#     保持与 x 相同的 device 和 dtype。
#     """
#     device = x.device
#     dtype = x.dtype
#     num_token, num_dims = x.size(-2), x.size(-1)
#     p = torch.zeros((1, num_token, num_dims), device=device, dtype=dtype)
#     t = torch.arange(num_token, dtype=dtype, device=device).reshape(-1, 1) / \
#         torch.pow(torch.tensor(1e4, dtype=dtype, device=device),
#                   torch.arange(0, num_dims, 2, dtype=dtype, device=device) / num_dims)
#     p[:, :, 0::2] = torch.sin(t)
#     p[:, :, 1::2] = torch.cos(t)
#     return p

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
    """Positional encoding.

    Defined in :numref:`sec_self-attention-and-positional-encoding`"""
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
    内部视作 (B, T, D) with D=F（frequency bins as feature dim）并在 time 维度做压缩 k_len。
    支持当输入 T != seq_len 时对 Pk/Pv 做线性插值重采样。
    返回与输入相同的形状 (B, F, T)（即把通道放回 frequency-first 形式）。
    """

    def __init__(self,
                 freq_bins: int,
                 att_dim: int,
                 groups: int,
                 seq_len: int,
                 k_len: int,
                 use_pos: bool = True,
                 dropout: float = 0.0,
                 device: str = 'cuda'):
        """
        freq_bins: F (feature dim)
        att_dim: query/key dim per group (d_a)
        groups: number of groups (must divide freq_bins)
        seq_len: reference time length used to create Pk/Pv
        k_len: compressed key/value length
        """
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
        self.device = device

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

        # 使用正弦-余弦位置编码（不再使用可学习 time_pos）
        # 保留 self.use_pos 开关，按需在 forward 中调用 position_coding(x)
        if use_pos:
            # (1, seq_len, 1) broadcastable for (B, T, D)
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
        # P.T shape -> (k_len, seq_len) -> make batch (1, k_len, seq_len) and interpolate along last dim
        Pt = P.t().unsqueeze(0)  # (1, k_len, seq_len)
        # F.interpolate supports 3D input for linear mode: (N, C, L)
        Pt_resized = F.interpolate(Pt, size=target_T, mode='linear', align_corners=False)  # (1, k_len, target_T)
        Pt_resized = Pt_resized.squeeze(0)  # (k_len, target_T)
        return Pt_resized.t()  # -> (target_T, k_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, F, T)
        returns: out (B, F, T)
        """

        B, C, F, T = x.shape
        x = x.squeeze(1)  # (B, F, T)
        # x = x.permute(0, 2, 1)  # (B, T, F)
        assert F == self.freq_bins, f"Input freq dim {F} must equal freq_bins {self.freq_bins}"

        # if self.use_pos:
        #     if T == self.seq_len:
        #         pos = self.time_pos  # (1, seq_len, 1)
        #     else:
        #         # resample time_pos to new T
        #         pos = F.interpolate(self.time_pos.permute(0,2,1), size=T, mode='linear', align_corners=False)  # (1,1,T)
        #         pos = pos.permute(0,2,1)  # (1, T, 1)
        #     # x shape (B, F, T) -> transpose to (B, T, F) then add pos (broadcast on F)
        #     x_t = x.permute(0, 2, 1) + pos  # (B, T, F)
        # else:
        #     x_t = x.permute(0, 2, 1)  # (B, T, F)

        # x_t = self.pos_encoding(x.permute(0, 2, 1))  # (B, T, F)

        # 位置编码：使用给定的正弦-余弦编码函数，直接按当前 T/F 生成，无需插值
        # x_t = self.encoding_to_sa(x)
        # if self.use_pos:
        #     pos = position_coding(x_t).to(self.device)  # (1, T, F)
        #     x_t = x_t + pos

        x_t = x
        # 注意：当前张量形状为 (B, F, T)，LayerNorm(freq) 需作用在 F 维。
        # 由于 LayerNorm 期望最后一维为特征维，所以这里先转置到 (B, T, F)，归一化后再保持该布局用于注意力计算。
        x_t = x_t.permute(0, 2, 1)  # (B, T, F)
        x_t = self.layer_norm(x_t)  # normalize over F

        B, T2, D = x_t.shape  # 现在 T2 == 原始 T, D == F
        G = self.groups
        gdim = self.gdim

        # split into groups along feature dim: shape -> (B, G, T, gdim)
        xg = x_t.view(B, T2, G, gdim).permute(0, 2, 1, 3)  # (B, G, T, gdim)

        outs = []
        for g in range(G):
            xg_g = xg[:, g]  # (B, T, gdim)
            q = self.qs[g](xg_g)  # (B, T, att_dim)
            k = self.ks[g](xg_g)  # (B, T, att_dim)
            v = self.vs[g](xg_g)  # (B, T, gdim)

            # resample Pk/Pv if T2 != seq_len（与当前序列长度对齐）
            Pk_g = self._resample_P(self.Pk[g], T2)  # (T2, k_len)
            Pv_g = self._resample_P(self.Pv[g], T2)  # (T2, k_len)

            # compress along time: einsum -> k_red: (B, k_len, att_dim), v_red: (B, k_len, gdim)
            k_red = torch.einsum('bta,tk->bka', k, Pk_g)  # (B, k_len, att_dim)
            v_red = torch.einsum('btg,tk->bkg', v, Pv_g)  # (B, k_len, gdim)

            # attention: (B,T,att_dim) @ (B,att_dim,k_len) -> (B,T,k_len)
            att = torch.matmul(q, k_red.transpose(1, 2)) * self.scale

            # 相对位置偏置：生成 (1, T2, k_len)，与 att 形状对齐并按 (B, T2, k_len) 广播
            bias = self.rel_pos_bias(T2, self.k_len).to(att.device)  # (1, T2, k_len)
            att = att + bias.squeeze(0)  # (T2, k_len)

            att = torch.softmax(att, dim=-1)
            att = self.dropout(att)

            # out: (B, T, gdim)
            out_g = torch.matmul(att, v_red)
            out_g = self.os[g](out_g)  # (B, T, gdim)
            outs.append(out_g)

        # concat groups along feature axis -> (B, T, D)
        out = torch.cat(outs, dim=-1)  # (B, T, F)

        # transpose back to (B, F, T)
        out = out.permute(0, 2, 1).contiguous()  # (B, F, T)
        return out


class RFProjectedAttentionClassifier(nn.Module):
    """
    将 GroupedProjectedAttentionRF 封装为一个分类器：
    输入 x: (B, F, T)
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
                 dropout: float = 0.1,
                 device='cuda'):
        super().__init__()
        self.att = GroupedProjectedAttentionRF(
            freq_bins=freq_bins,
            att_dim=att_dim,
            groups=groups,
            seq_len=seq_len,
            k_len=k_len,
            use_pos=use_pos,
            dropout=dropout,
            device=device
        )
        # a small conv block or linear bottleneck can be used; here用简单的 conv 做通道整合
        if hidden_dim is None:
            hidden_dim = max(128, freq_bins // 2)

        # after attention output (B, F, T)
        # we do global average pool over time and freq -> (B, F)
        self.pool = lambda x: x.mean(dim=-1).mean(dim=-1)  # (B,F) -> (B,)
        self.fc = nn.Sequential(
            nn.Linear(freq_bins, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, F, T)
        returns logits: (B, num_classes)
        """
        out = self.att(x)  # (B, F, T)
        # global average pooling over time and freq
        pooled = out.mean(dim=-1).mean(dim=-1)  # (B,)
        # careful: out.mean(dim=-1) -> (B, F); then mean dim=-1 -> (B,)
        # but we want (B, F) -> fc expects (B, F), so do:
        pooled = out.mean(dim=-1)  # (B, F)
        logits = self.fc(pooled)  # (B, num_classes)
        return logits



if False:
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    model = RFProjectedAttentionClassifier(freq_bins=128, seq_len=128, num_classes=25, groups=8, att_dim=32, k_len=64).to(device)

    x = torch.randn(16, 1, 128, 128).to(device)  # batch_size=16
    out = model(x)
    print(out.shape)

    from torchsummary import summary
    from thop import profile
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = RFProjectedAttentionClassifier(freq_bins=128, seq_len=128, num_classes=25, groups=8, att_dim=32, k_len=64)
    model = model.to(device)

    # summary(model, input_size=(1, 512))
    inputs = torch.randn(1, 1, 128, 128).to(device)
    flops, params = profile(model, (inputs,))
    print('flops: ', flops, 'params: ', params)
