import torch
import torch.nn as nn
from typing import Optional, Dict

# --------------------------------------------------------------------------
# 注意：这里假设 newmodels.Res_Inc_Net 和 newmodels.SpectraAtt_noProjected
# 已经存在于你的 Python 路径中，并且可以被导入。
# --------------------------------------------------------------------------
from newmodels.Res_Inc_Net import Res_Inc_Net_base
from newmodels.SpectraAtt_noProjected_downsample import GroupedProjectedAttentionRF


class ResInc_SpAtt_Finetune(nn.Module):
    """
    联合模型（特征提取/微调版）：
    加载预训练的 Res_Inc_Net 和 频谱注意力(SpectraAtt) 骨干，
    冻结两者，仅训练顶层的融合分类头。

    输入:  x (B, 1, F, T)，默认 (B, 1, 512, 512)
    流程:
      - [冻结] 分支A: Res_Inc_Net_base (加载预训练权重)
      - [冻结] 分支B: GroupedProjectedAttentionRF (加载预训练权重)
      - 拼接: [feat_res, feat_sp] -> (B, C_res + F)
      - [训练] 分类: MLP 头得到 logits (B, num_classes)
    """

    def __init__(
        self,
        # 权重路径
        resinc_weights_path: str,
        spectraatt_weights_path: str,
        # 模型配置（应与预训练模型一致）
        num_classes: int = 25,
        freq_bins: int = 512,
        seq_len: int = 128,
        groups: int = 8,
        att_dim: int = 32,
        k_len: int = 64, # k_len 在 GroupedProjectedAttentionRF 中似乎未用
        dropout: float = 0.1,
        hidden_dim: Optional[int] = 512,
    ) -> None:
        super().__init__()

        # -----------------------------------------------
        # 1. 初始化骨干网络
        # -----------------------------------------------

        # 分支A: 1D ResNet 骨干 (include_top=False)
        self.res_backbone = Res_Inc_Net_base(num_classes=num_classes, include_top=False)
        self.res_pool = nn.AdaptiveAvgPool1d(1)

        # 分支B: 频谱注意力 (GroupedProjectedAttentionRF)
        self.spectra_att = GroupedProjectedAttentionRF(
            freq_bins=freq_bins,
            att_dim=att_dim,
            groups=groups,
            seq_len=seq_len,
            use_pos=True, # 假设与预训练时一致
            dropout=dropout,
        )

        # -----------------------------------------------
        # 2. 加载预训练权重
        # -----------------------------------------------

        print(f"Loading ResIncNet weights from: {resinc_weights_path}")
        # 加载 ResIncNet 权重
        # 假设预训练模型是带 fc 头的 (include_top=True)
        # 使用 strict=False 来忽略 fc 层的权重（因为 self.res_backbone 没有 fc）
        res_state_dict = torch.load(resinc_weights_path, map_location="cpu")
        self.res_backbone.load_state_dict(res_state_dict, strict=False)


        print(f"Loading SpectraAtt weights from: {spectraatt_weights_path}")
        spec_full_state_dict = torch.load(spectraatt_weights_path, map_location="cpu")

        # 尝试剥离可能存在的 'att.' 前缀
        spec_backbone_state_dict = {}
        has_att_prefix = any(k.startswith('att.') for k in spec_full_state_dict.keys())
        if has_att_prefix:
            for k, v in spec_full_state_dict.items():
                if k.startswith('att.'):
                    spec_backbone_state_dict[k[len('att.'):]] = v
        else:
            # 直接使用（假设原本就是 GroupedProjectedAttentionRF 的 state_dict）
            spec_backbone_state_dict = spec_full_state_dict

        # 自适应处理相对位置偏置 (relative position bias) 尺寸不匹配问题
        current_state = self.spectra_att.state_dict()
        adjusted_state = {}
        skipped_keys = []
        from torch.nn import functional as F

        for k, v in spec_backbone_state_dict.items():
            if k not in current_state:
                continue  # 当前模型没有这个参数/缓冲
            cur_shape = current_state[k].shape
            if v.shape == cur_shape:
                adjusted_state[k] = v
            else:
                # 针对相对位置偏置表进行插值 (形状一般为 [num_heads, L_old] -> [num_heads, L_new])
                if 'rel_pos_bias.relative_bias' in k and v.dim() == 2 and current_state[k].dim() == 2:
                    old_len = v.shape[1]
                    new_len = cur_shape[1]
                    try:
                        # 线性插值到新长度
                        resized = F.interpolate(v.unsqueeze(0), size=new_len, mode='linear', align_corners=True).squeeze(0)
                        if resized.shape == cur_shape:
                            adjusted_state[k] = resized
                            print(f"[Info] Interpolated relative_bias from {v.shape} to {cur_shape}")
                            continue
                    except Exception as e:
                        print(f"[Warn] Interpolation failed for {k}: {e}")
                # rel_pos_index 一般为整数索引表，不强行覆盖，使用当前模型重新计算的
                if 'rel_pos_bias.rel_pos_index' in k:
                    print(f"[Info] Skip loading '{k}' (shape {v.shape} -> {cur_shape}); using model default.")
                    continue
                # 其他尺寸不匹配的直接跳过
                print(f"[Warn] Skip param '{k}' due to shape mismatch {v.shape} vs {cur_shape}")
                skipped_keys.append(k)

        missing_before = [k for k in current_state.keys() if k not in adjusted_state]
        # 合并要加载的权重
        load_status = self.spectra_att.load_state_dict(adjusted_state, strict=False)
        print(f"Loaded SpectraAtt weights with interpolation. Missing(keys not loaded or kept default): {len(missing_before)}")
        if skipped_keys:
            print(f"Skipped (shape mismatch) keys: {skipped_keys}")

        # -----------------------------------------------
        # 3. 冻结骨干网络
        # -----------------------------------------------
        print("Freezing ResIncNet backbone...")
        for param in self.res_backbone.parameters():
            param.requires_grad = False
        
        print("Freezing SpectraAtt backbone...")
        for param in self.spectra_att.parameters():
            param.requires_grad = False

        # -----------------------------------------------
        # 4. 初始化可训练的分类头
        # -----------------------------------------------
        C_res = 512  # 与 Res_Inc_Net_base 默认配置一致
        C_sp = freq_bins
        feat_dim = C_res + C_sp
        
        if hidden_dim is None:
            hidden_dim = max(256, feat_dim // 2)

        # 这个分类器默认 requires_grad=True，将会被训练
        self.classifier = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, num_classes),
        )
        print("Classifier head is ready for training.")


    def forward(self, x: torch.Tensor, return_features: bool = False) -> torch.Tensor:
        """
        前向传播。骨干网络在此处以 .eval() 模式运行（如果已调用 model.eval()）
        并处于 no_grad 上下文中（如果用于推理）。
        """
        # 分支A: Res_Inc_Net (已冻结)
        res_feat_map = self.res_backbone(x)
        res_feat = self.res_pool(res_feat_map).squeeze(-1)

        # 分支B: 频谱注意力 (已冻结)
        sp_map = self.spectra_att(x)
        sp_feat = sp_map.mean(dim=-1)

        # 拼接特征
        feat = torch.cat([res_feat, sp_feat], dim=-1)

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
    工厂函数：创建用于微调的 ResInc_SpAtt_Finetune 模型。
    
    必须提供预训练权重路径。
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
    # 示例：如何使用这个微调模型
    # =======================================================
    
    # 假设这是你的预训练权重文件
    RES_WEIGHTS = "/home/kk/lzk/SRSpecNet/1110/Res_Inc_Net_base/model.pth"
    ATT_WEIGHTS = "/home/kk/lzk/SRSpecNet/1110/rf_att_medium/model.pth" # 假设你训练的是 medium 版

    # 检查文件是否存在（仅为示例）
    import os
    if not (os.path.exists(RES_WEIGHTS) and os.path.exists(ATT_WEIGHTS)):
        print("="*50)
        print(f"警告：找不到示例权重路径：\n{RES_WEIGHTS}\n{ATT_WEIGHTS}")
        print("请将 'RES_WEIGHTS' 和 'ATT_WEIGHTS' 替换为你自己的实际路径")
        print("跳过模型实例化和优化器测试...")
        print("="*50)
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # 1. 实例化模型，传入权重路径
        model = ResInc_SpAtt(
            num_classes=10, # 假设你的最终任务是10分类
            resinc_weights_path=RES_WEIGHTS,
            spectraatt_weights_path=ATT_WEIGHTS,
            # 其他参数应与预训练时匹配
            freq_bins=512,
            seq_len=512, # 假设输入 T=512
            att_dim=32,  # 假设是 medium 版
        ).to(device)

        # 2. 创建优化器
        # 关键：filter(lambda p: p.requires_grad, model.parameters())
        # 这将自动只选择那些 'requires_grad=True' 的参数（即只有分类头的参数）
        
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=1e-3
        )

        # 验证：打印可训练参数的名称
        print("\n--- 可训练的参数 (应该只有 'classifier' 相关) ---")
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name)
        print("--------------------------------------------------")

        # 3. 模拟训练
        model.train() # 即使骨干被冻结，也设置为 train 模式以启用 Dropout
        
        dummy_input = torch.randn(4, 1, 512, 512).to(device) # (B, C, F, T)
        labels = torch.randint(0, 10, (4,)).to(device)
        
        optimizer.zero_grad()
        logits = model(dummy_input)
        
        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits, labels)
        
        loss.backward()
        optimizer.step()
        
        print(f"\n模拟训练完成。 Loss: {loss.item()}")
        
        # 4. 模拟评估
        model.eval()
        with torch.no_grad():
            logits_eval = model(dummy_input)
            print(f"模拟评估完成。 Logits shape: {logits_eval.shape}")