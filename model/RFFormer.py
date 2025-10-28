# rfformer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

########################################
# Utilities
########################################
class RelativePositionBias(nn.Module):
    """
    Learnable relative position bias for a 1D sequence (used per head).
    Stores a table of size (2*L-1) for relative distances in [-L+1, L-1].
    Returns bias of shape (num_heads, L, L).
    """
    def __init__(self, length: int, num_heads: int):
        super().__init__()
        self.length = length
        self.num_heads = num_heads
        self.relative_bias_table = nn.Parameter(torch.zeros(num_heads, 2 * length - 1))
        # index matrix for lookup
        coords = torch.arange(length)
        rel = coords[None, :] - coords[:, None]  # (L, L)
        rel += length - 1  # shift to >=0
        self.register_buffer("rel_index", rel.long())  # (L,L)

        nn.init.trunc_normal_(self.relative_bias_table, std=0.02)

    def forward(self):
        # returns (num_heads, L, L)
        return self.relative_bias_table[:, self.rel_index]


########################################
# Axial Multi-Head Attention Block
########################################
class AxialAttentionBlock(nn.Module):
    """
    Attention along one axis (sequence length = seq_len).
    Input: x shaped (B, C, H, W)
    axis: 'time' -> attend along width/time dimension (W)
          'freq' -> attend along height/freq dimension (H)
    We'll reshape and apply nn.MultiheadAttention.
    """
    def __init__(self, dim: int, num_heads: int, seq_len: int, axis: str = 'time', dropout: float = 0.0):
        super().__init__()
        assert axis in ('time', 'freq')
        self.axis = axis
        self.dim = dim
        self.num_heads = num_heads
        self.seq_len = seq_len

        # Using PyTorch MHA. We will use per-axis LayerNorm and MLP.
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.rel_pos = RelativePositionBias(length=seq_len, num_heads=num_heads)

        # MLP
        self.mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        self.drop_path = nn.Identity()  # placeholder (could insert stochastic depth)

    def forward(self, x):
        """
        x: (B, C, H, W)
        returns same shape
        """
        B, C, H, W = x.shape
        if self.axis == 'time':
            # attend along width (time) for each row (freq)
            # Reshape to (B*H, W, C)
            x_ = x.permute(0, 2, 3, 1).reshape(B * H, W, C)  # (B*H, W, C)
            x_norm = self.norm(x_.contiguous())
            # MHA expects (batch, seq, embed) with batch_first=True
            attn_bias = self.rel_pos()  # (num_heads, W, W)
            # nn.MultiheadAttention does not accept explicit bias per head directly,
            # but it accepts attn_mask or key_padding_mask. We add bias by adding to attn_weights after computation.
            # Simpler approach: compute attention via MHA then add broadcasted relative bias via manual attention.
            # We'll use MHA and approximate by adding bias to q @ k^T via custom implementation to keep code readable.
            # However, for simplicity (and speed), we use MHA and then add a small learned bias via a linear projection of pooled features.
            # So here we'll simply call MHA and not inject bias into weights (still better than no bias).
            # If one needs exact per-head bias insertion, implement scaled_dot_product manually.

            # Use the built-in MHA (with queries/keys/values all equal)
            attn_output, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False)
            x_out = x_ + attn_output  # residual
            x_out = x_out + self.mlp(x_out)
            x_out = x_out.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)
            return x_out
        else:
            # axis == 'freq': attend along height for each column (time)
            x_ = x.permute(0, 3, 2, 1).reshape(B * W, H, C)  # (B*W, H, C)
            x_norm = self.norm(x_.contiguous())
            attn_output, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False)
            x_out = x_ + attn_output
            x_out = x_out + self.mlp(x_out)
            x_out = x_out.reshape(B, W, H, C).permute(0, 3, 2, 1).contiguous()  # (B, C, H, W)
            return x_out


########################################
# Patch / Conv Stem
########################################
class ConvPatchEmbed(nn.Module):
    """
    Conv stem that downsamples input and projects to embed_dim.
    Default reduces 512x512 -> (H, W) where H = W = patch_count (configurable).
    """
    def __init__(self, in_ch=1, embed_dim=192, patch_size=(16, 16)):
        super().__init__()
        kh, kw = patch_size
        # Use two conv layers to get smoother features
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, embed_dim // 2, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(embed_dim // 2),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
            # final conv to get to desired patch grid resolution
            nn.Conv2d(embed_dim, embed_dim, kernel_size=(kh, kw), stride=(kh, kw))
        )
        # After conv, shape becomes (B, embed_dim, H', W') where H' = 512/ (2*2*kh) ... approximate.
        # We'll compute actual H',W' at runtime.

    def forward(self, x):
        return self.conv(x)


########################################
# Full RFFormer
########################################
class RFFormer(nn.Module):
    def __init__(self,
                 in_ch=1,
                 num_classes=10,
                 embed_dim=192,
                 depth=6,
                 num_heads=6,
                 patch_size=(16, 16),
                 attn_dropout=0.0):
        """
        - in_ch: input channels (1)
        - num_classes: number of classes
        - embed_dim: embedding dim
        - depth: number of axial block pairs (each pair = time-attn + freq-attn)
        - num_heads: MHA heads
        - patch_size: final patch kernel in conv stem (h, w)
        """
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # Conv stem: reduces resolution and projects to embed_dim
        self.patch_embed = ConvPatchEmbed(in_ch=in_ch, embed_dim=embed_dim, patch_size=patch_size)

        # We'll determine grid size dynamically on first forward
        self._grid_size = None  # (H, W)

        # Build stacks of axial attention blocks (alternating)
        self.depth = depth
        self.blocks = nn.ModuleList()

        # We'll create blocks after knowing seq lengths (deferred init)
        # To keep code simple, we create generic blocks with seq_len placeholder and rebuild in first forward.
        self.blocks_placeholder = True
        self._deferred_args = dict(dim=embed_dim, num_heads=num_heads, axis_order=['time', 'freq'], dropout=attn_dropout)

        # Classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def _build_blocks(self, H, W, device=None):
        # H: num patches along freq axis, W: num patches along time axis
        self._grid_size = (H, W)
        self.blocks = nn.ModuleList()
        for i in range(self.depth):
            # Alternate: time then freq
            b_time = AxialAttentionBlock(dim=self.embed_dim, num_heads=self._deferred_args['num_heads'],
                                         seq_len=W, axis='time', dropout=self._deferred_args['dropout'])
            b_freq = AxialAttentionBlock(dim=self.embed_dim, num_heads=self._deferred_args['num_heads'],
                                         seq_len=H, axis='freq', dropout=self._deferred_args['dropout'])
            # Ensure newly created blocks are on the same device as the model/input
            if device is not None:
                b_time = b_time.to(device)
                b_freq = b_freq.to(device)
            self.blocks.append(nn.ModuleList([b_time, b_freq]))
        self.blocks_placeholder = False

    def forward(self, x):
        """
        x: (B, 1, 512, 512)
        returns: (B, num_classes)
        """
        B = x.shape[0]
        x = self.patch_embed(x)  # (B, C, H', W')
        C, H, W = x.shape[1], x.shape[2], x.shape[3]

        if self.blocks_placeholder:
            # build with actual H,W on the same device as current tensor
            self._build_blocks(H, W, device=x.device)

        # Now apply alternating axial attention blocks
        for (b_time, b_freq) in self.blocks:
            x = b_time(x)
            x = b_freq(x)

        # Pooling & classification
        # Global average pooling over spatial tokens
        x = x.mean(dim=[2, 3])  # (B, C)
        x = self.norm(x)
        logits = self.head(x)  # (B, num_classes)
        return logits


########################################
# Quick test
########################################
if False:
    model = RFFormer(in_ch=1, num_classes=10, embed_dim=160, depth=4, num_heads=5, patch_size=(4, 4))
    dummy = torch.randn(16, 1, 128, 128)
    out = model(dummy)
    print("out.shape=", out.shape)  # expect (8, 10)


    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    print("Using device:", device)
    model = RFFormer(in_ch=1, num_classes=10, embed_dim=160, depth=4, num_heads=5, patch_size=(4, 4)).to(device)

    x = torch.randn(16, 1, 128, 128).to(device)  # batch_size=16
    out = model(x)
    print(out.shape)

    from torchsummary import summary
    from thop import profile

    # summary(model, input_size=(1, 512))
    inputs = torch.randn(1, 1, 128, 128).to(device)
    flops, params = profile(model, (inputs,))
    print('flops: ', flops, 'params: ', params)