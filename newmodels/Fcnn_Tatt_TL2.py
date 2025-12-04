import torch
import torch.nn as nn
from typing import Optional, Dict

# --------------------------------------------------------------------------
# 假设 newmodels.Res_Inc_Net 和 newmodels.SpectraAtt_noProjected 可被导入
# --------------------------------------------------------------------------
from newmodels.Res_Inc_Net import Res_Inc_Net_base
from newmodels.SpectraAtt_noProjected import GroupedProjectedAttentionRF
# 确保 F.interpolate 可用
from torch.nn import functional as F


class ResInc_SpAtt_Finetune(nn.Module):
    """
    联合模型（动态加权微调版）：
    加载预训练骨干，冻结两者，仅训练顶层的“动态门控”融合头。

    流程:
      - [冻结] 分支A (ResNet) -> feat_res (B, 512)
      - [冻结] 分支B (SpectraAtt) -> feat_sp (B, 512)
      - [训练] 门控网络(feat_res + feat_sp) -> weights (B, 2)
      - 融合: feat = weights[:,0] * feat_res + weights[:,1] * feat_sp
      - [训练] 分类: MLP(feat) -> logits (B, num_classes)
    """

    def __init__(
        self,
        # 权重路径
        resinc_weights_path: str,
        spectraatt_weights_path: str,
        # 模型配置
        num_classes: int = 25,
        freq_bins: int = 512,
        seq_len: int = 128,
        groups: int = 8,
        att_dim: int = 32,
        k_len: int = 64,
        dropout: float = 0.1,
        hidden_dim: Optional[int] = 512, # 注意：此参数在当前分类器结构中未被使用
    ) -> None:
        super().__init__()

        # -----------------------------------------------
        # 1. 初始化骨干网络 (与之前相同)
        # -----------------------------------------------
        self.res_backbone = Res_Inc_Net_base(num_classes=num_classes, include_top=False)
        self.res_pool = nn.AdaptiveAvgPool1d(1)

        self.spectra_att = GroupedProjectedAttentionRF(
            freq_bins=freq_bins,
            att_dim=att_dim,
            groups=groups,
            seq_len=seq_len,
            use_pos=True,
            dropout=dropout,
        )

        # -----------------------------------------------
        # 2. 加载预训练权重 (与之前相同)
        # -----------------------------------------------
        # (此处省略了加载和插值权重的代码，与你提供的版本保持一致)
        print(f"Loading ResIncNet weights from: {resinc_weights_path}")
        res_state_dict = torch.load(resinc_weights_path, map_location="cpu")
        self.res_backbone.load_state_dict(res_state_dict, strict=False)

        print(f"Loading SpectraAtt weights from: {spectraatt_weights_path}")
        spec_full_state_dict = torch.load(spectraatt_weights_path, map_location="cpu")

        spec_backbone_state_dict = {}
        has_att_prefix = any(k.startswith('att.') for k in spec_full_state_dict.keys())
        if has_att_prefix:
            for k, v in spec_full_state_dict.items():
                if k.startswith('att.'):
                    spec_backbone_state_dict[k[len('att.'):]] = v
        else:
            spec_backbone_state_dict = spec_full_state_dict

        current_state = self.spectra_att.state_dict()
        adjusted_state = {}
        skipped_keys = []
        for k, v in spec_backbone_state_dict.items():
            if k not in current_state: continue
            cur_shape = current_state[k].shape
            if v.shape == cur_shape:
                adjusted_state[k] = v
            else:
                if 'rel_pos_bias.relative_bias' in k and v.dim() == 2 and current_state[k].dim() == 2:
                    old_len, new_len = v.shape[1], cur_shape[1]
                    try:
                        resized = F.interpolate(v.unsqueeze(0), size=new_len, mode='linear', align_corners=True).squeeze(0)
                        if resized.shape == cur_shape:
                            adjusted_state[k] = resized
                            print(f"[Info] Interpolated relative_bias from {v.shape} to {cur_shape}")
                            continue
                    except Exception as e:
                        print(f"[Warn] Interpolation failed for {k}: {e}")
                if 'rel_pos_bias.rel_pos_index' in k:
                    print(f"[Info] Skip loading '{k}' (shape {v.shape} -> {cur_shape}); using model default.")
                    continue
                print(f"[Warn] Skip param '{k}' due to shape mismatch {v.shape} vs {cur_shape}")
                skipped_keys.append(k)

        missing_before = [k for k in current_state.keys() if k not in adjusted_state]
        load_status = self.spectra_att.load_state_dict(adjusted_state, strict=False)
        print(f"Loaded SpectraAtt weights with interpolation. Missing(keys not loaded or kept default): {len(missing_before)}")

        # -----------------------------------------------
        # 3. 冻结骨干网络 (与之前相同)
        # -----------------------------------------------
        print("Freezing ResIncNet backbone...")
        for param in self.res_backbone.parameters():
            param.requires_grad = False
        
        print("Freezing SpectraAtt backbone...")
        for param in self.spectra_att.parameters():
            param.requires_grad = False

        # -----------------------------------------------
        # 4. 初始化可训练的分类头 (!!! 修改点 !!!)
        # -----------------------------------------------
        C_res = 512  # 与 Res_Inc_Net_base 默认配置一致
        C_sp = freq_bins

        # 确保两个分支维度相同
        if C_res != C_sp:
            raise ValueError(f"动态加权 (Gating) 要求特征维度相同, "
                             f"但 ResIncNet (C_res={C_res}) 和 SpectraAtt (C_sp={C_sp}) 维度不同")
        self.common_dim = C_res

        # 新增：门控网络 (可训练)
        # 它接收两个特征的和，并输出2个权重
        gate_input_dim = self.common_dim
        self.gating_mlp = nn.Sequential(
            nn.Linear(gate_input_dim, 32), # 缩减维度
            nn.ReLU(inplace=True),
            nn.Linear(32, 2),              # 2个分支的权重
            nn.Softmax(dim=-1)             # 归一化
        )

        # 修改：分类器现在接收 common_dim (512)，而不是 C_res + C_sp (1024)
        feat_dim = self.common_dim*2
        
        # 这个分类器默认 requires_grad=True，将会被训练
        self.classifier = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, num_classes),
        )
        print("Classifier head (Gating) is ready for training.")


    def forward(self, x: torch.Tensor, return_features: bool = False) -> torch.Tensor:
        """
        前向传播 (!!! 修改点 !!!)
        """
        # 分支A: Res_Inc_Net (已冻结)
        res_feat_map = self.res_backbone(x)
        res_feat = self.res_pool(res_feat_map).squeeze(-1) # (B, 512)

        # 分支B: 频谱注意力 (已冻结)
        sp_map = self.spectra_att(x)
        sp_feat = sp_map.mean(dim=-1) # (B, 512)

        # # --- 特征融合 (从 cat 替换为 动态加权) ---
        
        # # 1. 使用 (res_feat + sp_feat) 作为门控网络的输入
        # gate_input = res_feat + sp_feat
            
        # # 2. 计算权重
        # weights = self.gating_mlp(gate_input) # -> (B, 2)
        
        # # 3. 加权求和
        # # weights[:, 0] -> res_feat 的权重
        # # weights[:, 1] -> sp_feat 的权重
        # w1 = weights[:, 0].unsqueeze(1) # (B, 1)
        # w2 = weights[:, 1].unsqueeze(1) # (B, 1)
        feat_sum = res_feat + sp_feat
        feat_prod = res_feat * sp_feat
        
        # 广播: (B, 1) * (B, 512) + (B, 1) * (B, 512)
        feat = torch.cat([feat_sum, feat_prod], dim=-1) # 最终特征 (B, 512)

        # -------------------------------------------

        if return_features:
            return feat

        # 分类头 (可训练)
        logits = self.classifier(feat)
        return logits


def ResInc_SpAtt(num_classes: int = 25,
                 resinc_weights_path: str = None,
                 spectraatt_weights_path: str = None,
                 freq_bins: int = 512,
                 seq_len: int = 128,
                 groups: int = 8,
                 att_dim: int = 32,
                 k_len: int = 64,
                 dropout: float = 0.1,
                 hidden_dim: Optional[int] = 512):
    """
    工厂函数 (无需修改)
    """
    if resinc_weights_path is None or spectraatt_weights_path is None:
        raise ValueError("必须为微调模型提供 'resinc_weights_path' 和 'spectraatt_weights_path'")
        
    return ResInc_SpAtt_Finetune(
        resinc_weights_path=resinc_weights_path,
        spectraatt_weights_path=spectraatt_weights_path,
        num_classes=num_classes,
        freq_bins=freq_bins,
        seq_len=seq_len,
        groups=groups,
        att_dim=att_dim,
        k_len=k_len,
        dropout=dropout,
        hidden_dim=hidden_dim,
    )

if False:
    # =======================================================
    # 示例 (!!! 修改点 !!!)
    # =======================================================
    
    # (假设你的路径)
    RES_WEIGHTS = "/home/kk/lzk/SRSpecNet/1110/Res_Inc_Net_base/model.pth"
    ATT_WEIGHTS = "/home/kk/lzk/SRSpecNet/1110/rf_att_medium/model.pth"

    import os
    if not (os.path.exists(RES_WEIGHTS) and os.path.exists(ATT_WEIGHTS)):
        print("="*50)
        print(f"警告：找不到示例权重路径")
        print("="*50)
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # 1. 实例化模型 (与之前相同)
        model = ResInc_SpAtt(
            num_classes=10, 
            resinc_weights_path=RES_WEIGHTS,
            spectraatt_weights_path=ATT_WEIGHTS,
            freq_bins=512,
            seq_len=512, 
            att_dim=32,
        ).to(device)

        # 2. 创建优化器 (与之前相同)
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=1e-3
        )

        # 验证：打印可训练参数的名称 (!!! 修改点 !!!)
        print("\n--- 可训练的参数 (Gating版: classifier + gating_mlp) ---")
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name)
        print("--------------------------------------------------")
        
        # 预期输出:
        # gating_mlp.0.weight
        # gating_mlp.0.bias
        # gating_mlp.2.weight
        # gating_mlp.2.bias
        # classifier.0.weight
        # classifier.0.bias
        # classifier.2.weight
        # classifier.2.bias
        
        # 3. 模拟训练 (与之前相同)
        model.train() 
        dummy_input = torch.randn(4, 1, 512, 512).to(device)
        labels = torch.randint(0, 10, (4,)).to(device)
        optimizer.zero_grad()
        logits = model(dummy_input)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        print(f"\n模拟训练完成。 Loss: {loss.item()}")
        
        # 4. 模拟评估 (与之前相同)
        model.eval()
        with torch.no_grad():
            logits_eval = model(dummy_input)
            print(f"模拟评估完成。 Logits shape: {logits_eval.shape}")