import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

# ==========================================================================================
# Code from Res_Inc_Net.py
# ==========================================================================================

class BasicBlock1D(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, groups=1, **kwargs):
        super(BasicBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False, groups=groups)
        self.bn1 = nn.BatchNorm1d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
        self.bn2 = nn.BatchNorm1d(out_channel)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


class ResNet1D(nn.Module):
    def __init__(self,
                 block,
                 blocks_num,
                 num_classes=1000,
                 include_top=True,
                 groups=1,
                 width_per_group=64):
        super(ResNet1D, self).__init__()
        self.include_top = include_top
        self.in_channel = 64
        self.groups = groups
        self.width_per_group = width_per_group

        # Input channel is 1
        self.conv1 = nn.Conv1d(1, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool1d(1)  # output size = 1
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channel, channel * block.expansion, kernel_size=1,
                          stride=stride, bias=False, groups=self.groups),
                nn.BatchNorm1d(channel * block.expansion))

        layers = []
        layers.append(block(self.in_channel,
                            channel,
                            downsample=downsample,
                            stride=stride,
                            groups=self.groups,
                            width_per_group=self.width_per_group))
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channel,
                                channel,
                                groups=self.groups,
                                width_per_group=self.width_per_group))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

        return x

def Res_Inc_Net(num_classes=1000, include_top=True):
    # Using ResNet1D with BasicBlock1D and [1, 1, 1, 1] blocks
    return ResNet1D(BasicBlock1D, [1, 1, 1, 1], num_classes=num_classes, include_top=include_top, groups=32, width_per_group=4)

# ==========================================================================================
# Code from RFProjectedAttentionClassifier.py
# ==========================================================================================

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
        if k_len is None:
            k_len = q_len
        if q_len == k_len == self.seq_len:
            return self.relative_bias[:, self.rel_pos_index]
        
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

class GroupedProjectedAttentionRF(nn.Module):
    def __init__(self,
                 freq_bins: int,
                 att_dim: int,
                 groups: int,
                 seq_len: int,
                 k_len: int,
                 use_pos: bool = True,
                 dropout: float = 0.0,
                 device: str = 'cuda'):
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
        
        self.qs = nn.ModuleList([nn.Linear(self.gdim, att_dim, bias=False) for _ in range(groups)])
        self.ks = nn.ModuleList([nn.Linear(self.gdim, att_dim, bias=False) for _ in range(groups)])
        self.vs = nn.ModuleList([nn.Linear(self.gdim, self.gdim, bias=False) for _ in range(groups)])
        self.os = nn.ModuleList([nn.Linear(self.gdim, self.gdim, bias=False) for _ in range(groups)])

        self.Pk = nn.ParameterList([nn.Parameter(torch.randn(seq_len, k_len) * (1.0 / math.sqrt(seq_len))) for _ in range(groups)])
        self.Pv = nn.ParameterList([nn.Parameter(torch.randn(seq_len, k_len) * (1.0 / math.sqrt(seq_len))) for _ in range(groups)])

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.layer_norm = nn.LayerNorm(freq_bins)
        self.rel_pos_bias = RelativePositionBias(seq_len=seq_len, num_heads=1)

    def _resample_P(self, P: torch.Tensor, target_T: int) -> torch.Tensor:
        if target_T == self.seq_len:
            return P
        Pt = P.t().unsqueeze(0)
        Pt_resized = F.interpolate(Pt, size=target_T, mode='linear', align_corners=False)
        return Pt_resized.squeeze(0).t()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expects x: (B, 1, F, T)
        B, C, F, T = x.shape
        x = x.squeeze(1)  # -> (B, F, T)
        assert F == self.freq_bins, f"Input freq dim {F} must equal freq_bins {self.freq_bins}"

        # x_t = x.permute(0, 2, 1)  # -> (B, T, F)
        x_t = x
        x_t = self.layer_norm(x_t)

        B, T2, D = x_t.shape
        G = self.groups
        gdim = self.gdim

        xg = x_t.view(B, T2, G, gdim).permute(0, 2, 1, 3)

        outs = []
        for g in range(G):
            xg_g = xg[:, g]
            q = self.qs[g](xg_g)
            k = self.ks[g](xg_g)
            v = self.vs[g](xg_g)

            Pk_g = self._resample_P(self.Pk[g], T2)
            Pv_g = self._resample_P(self.Pv[g], T2)

            k_red = torch.einsum('bta,tk->bka', k, Pk_g)
            v_red = torch.einsum('btg,tk->bkg', v, Pv_g)

            att = torch.matmul(q, k_red.transpose(1, 2)) * self.scale
            bias = self.rel_pos_bias(T2, self.k_len).to(att.device)
            att = att + bias
            att = torch.softmax(att, dim=-1)
            att = self.dropout(att)

            out_g = torch.matmul(att, v_red)
            out_g = self.os[g](out_g)
            outs.append(out_g)

        out = torch.cat(outs, dim=-1)
        out = out.permute(0, 2, 1).contiguous()  # -> (B, F, T)
        return out

# ==========================================================================================
# New Combined Model
# ==========================================================================================

class ResInc_GAtt(nn.Module):
    """
    A combined model that uses both ResNet1D and Projected Attention
    to extract features, which are then fused for final classification.
    """
    def __init__(self, num_classes=10, dropout_rate=0.0):
        super().__init__()
        
        # --- Branch 1: ResNet1D based feature extractor ---
        # We instantiate Res_Inc_Net without its final classification layer.
        self.resnet_features = Res_Inc_Net(num_classes=num_classes, include_top=False)
        
        # --- Branch 2: Projected Attention based feature extractor ---
        # We instantiate the core attention module.
        # Input dimensions (F=512, T=512) are set here.
        self.attention_features = GroupedProjectedAttentionRF(
            freq_bins=512,
            seq_len=512,  # Reference sequence length
            att_dim=32,   # Attention dimension per group
            groups=8,     # Number of attention groups
            k_len=64,     # Compressed key/value length
            use_pos=True,
            dropout=0.1,
        )
        
        # --- Fusion and Classification Head ---
        # The ResNet branch outputs a 512-dim vector after pooling.
        # The Attention branch outputs a 512-dim vector after pooling.
        # Concatenated feature dimension is 512 + 512 = 1024.
        self.fusion_classifier = nn.Sequential(
            nn.Linear(512 + 512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the combined model.
        Args:
            x (torch.Tensor): Input tensor of shape (B, 1, 512, 512).
        Returns:
            torch.Tensor: Logits of shape (B, num_classes).
        """
        # Input validation
        if x.dim() != 4 or x.shape[1] != 1 or x.shape[2] != 512 or x.shape[3] != 512:
            raise ValueError(f"Expected input shape (B, 1, 512, 512), but got {x.shape}")

        # --- Branch 1: ResNet ---
        # Prepare input for ResNet1D: (B, 1, 512, 512) -> (B, 1, 512) by averaging the last dim.
        x_res = x.mean(dim=-1)
        
        # Extract features: output shape (B, 512, 16)
        res_out = self.resnet_features(x_res)
        
        # Global average pooling to get a fixed-size feature vector: (B, 512, 16) -> (B, 512)
        res_features = F.adaptive_avg_pool1d(res_out, 1).squeeze(-1)
        
        # --- Branch 2: Attention ---
        # The attention module expects (B, 1, 512, 512), so we can pass 'x' directly.
        
        # Extract features: output shape (B, 512, 512)
        att_out = self.attention_features(x)
        
        # Global average pooling over the time dimension: (B, 512, 512) -> (B, 512)
        att_features = att_out.mean(dim=-1)
        
        # --- Fusion and Classification ---
        # Concatenate features from both branches along the feature dimension.
        fused_features = torch.cat((res_features, att_features), dim=1) # Shape: (B, 1024)
        
        # Pass through the final classifier to get logits.
        logits = self.fusion_classifier(fused_features) # Shape: (B, num_classes)
        
        return logits


# ==========================================================================================
# Example Usage
# ==========================================================================================
if __name__ == '__main__':
    # # Define model parameters
    # BATCH_SIZE = 4
    # NUM_CLASSES = 10
    
    # # Check for GPU availability
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(f"Using device: {device}")
    
    # # Instantiate the model
    # model = ResInc_GAtt(num_classes=NUM_CLASSES)
    # model.to(device)
    
    # # Create a dummy input tensor with the specified dimensions
    # # Shape: (batch_size, channels, frequency_bins, time_steps)
    # dummy_input = torch.randn(BATCH_SIZE, 1, 512, 512).to(device)
    
    # print(f"Input shape: {dummy_input.shape}")
    
    # # Perform a forward pass
    # try:
    #     output_logits = model(dummy_input)
    #     print(f"Output shape: {output_logits.shape}")
        
    #     # Verify output shape
    #     assert output_logits.shape == (BATCH_SIZE, NUM_CLASSES)
    #     print("\nModel forward pass successful and output shape is correct.")
        
    # except Exception as e:
    #     print(f"\nAn error occurred during the forward pass: {e}")

    from torchsummary import summary
    from thop import profile
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = ResInc_GAtt(num_classes=10)
    model = model.to(device)

    # summary(model, input_size=(1, 512))
    inputs = torch.randn(1, 1, 512, 512).to(device)
    flops, params = profile(model, (inputs,))
    print('flops: ', flops, 'params: ', params)
