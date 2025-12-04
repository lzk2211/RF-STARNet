#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
HybridRFClassifier.py

融合模型文件。
该模型导入 ResNet1D (来自 Res_Inc_Net_SWConv_SENet.py) 和 
RFProjectedAttention (来自 SpectraAtt_noProjected_downsample.py) 
作为两个并行的特征提取分支。

它将两个分支的特征拼接 (concatenate) 起来，
并使用一个小型的、可训练的分类头进行最终分类。

设计用于迁移学习，两个骨干网络可以被冻结。
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

# 
# 关键步骤：从您现有的 .py 文件中导入模型类和工厂函数
# 
try:
    # 导入分支 A (ResNet)
    # 我们使用 SW_Res_Inc_Net_base 作为示例，它返回一个 include_top=False 的 ResNet1D
    from newmodels.Res_Inc_Net_SWConv_SENet import SW_Res_Inc_Net_base
    
    # 导入分支 B (SpectraAtt)
    # 我们使用 rf_att_medium 来构建模型并"窃取"它的主干
    from newmodels.SpectraAtt_noProjected_downsample import rf_att_medium

except ImportError as e:
    print("未能在 'newmodels' 文件夹下找到模块，尝试从当前目录导入...")
    from Res_Inc_Net_SWConv_SENet import SW_Res_Inc_Net_base
    from SpectraAtt_noProjected_downsample import rf_att_medium

class HybridRFClassifier(nn.Module):
    """
    融合频谱注意力 (SpectraAtt) 和 1D-ResNet (Res_Inc_Net) 的混合分类器。
    
    ### 修改：权重加载已移入 __init__ ###
    """
    def __init__(self,
                 num_classes: int,
                 res_feat_dim: int = 512,
                 spectra_feat_dim: int = 512,
                 freq_bins: int = 512,
                 seq_len: int = 512,
                 stem_pool_factor: int = 8, 
                 n_div: int = 4,
                 # --- 新增参数：用于在初始化时加载权重 ---
                 res_weights_path: Optional[str] = None,
                 spectra_weights_path: Optional[str] = None,
                 device: str = "cpu"
                 ):
        super().__init__()
        self.num_classes = num_classes
        self.res_feat_dim = res_feat_dim
        self.spectra_feat_dim = spectra_feat_dim

        # --- 1. 初始化分支 A (ResNet) ---
        self.res_backbone = SW_Res_Inc_Net_base(
            num_classes=num_classes, 
            include_top=False, 
            n_div=n_div
        )
        self.res_pool = nn.AdaptiveAvgPool1d(1)
        
        # --- 2. 初始化分支 B (SpectraAtt) ---
        self.spectra_att_full_model = rf_att_medium(
            freq_bins=freq_bins,
            seq_len=seq_len,
            num_classes=num_classes,
            stem_pool_factor=stem_pool_factor
        )
        # 提取所需的模块
        self.spectra_stem = self.spectra_att_full_model.stem_pool
        self.spectra_att_block = self.spectra_att_full_model.att
        
        # --- 3. 初始化可训练的分类头 ---
        self.fused_dim = self.res_feat_dim + self.spectra_feat_dim
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.fused_dim),
            nn.Linear(self.fused_dim, num_classes),
        )

        # --- 4. (新增) 在 __init__ 中加载权重 ---
        
        # 加载 ResNet 权重
        if res_weights_path:
            try:
                print(f"正在从 {res_weights_path} 加载 ResNet 骨干权重...")
                resnet_weights = torch.load(res_weights_path, map_location=device)
                
                # 使用 strict=False 忽略 SE vs ECA 的不匹配 和 fc 层的缺失
                self.res_backbone.load_state_dict(resnet_weights, strict=False)
                
                print("ResNet 权重加载完成 (strict=False)。")
                print("警告：已忽略 'fc' 层和任何不匹配的注意力键 (SE vs ECA)。")

            except FileNotFoundError:
                print(f"警告: 未找到 ResNet 权重文件 {res_weights_path}")
            except Exception as e:
                print(f"加载 ResNet 权重时出错: {e}")

        # 加载 SpectraAtt 权重
        if spectra_weights_path:
            try:
                print(f"正在从 {spectra_weights_path} 加载 SpectraAtt 骨干权重...")
                spectra_weights_dict = torch.load(spectra_weights_path, map_location=device)
                
                # --- ### 关键修改 ### ---
                # 
                # 手动删除因 seq_len 变化而导致“形状不匹配”的键
                # 
                keys_to_remove = [
                    "att.rel_pos_bias.relative_bias", 
                    "att.rel_pos_bias.rel_pos_index"
                ]
                
                removed_keys = []
                for key in keys_to_remove:
                    if key in spectra_weights_dict:
                        spectra_weights_dict.pop(key)
                        removed_keys.append(key)
                
                if removed_keys:
                    print(f"警告: 已从权重中移除 {len(removed_keys)} 个形状不匹配的键 (rel_pos_bias)。")
                    print("     这些层将保持随机初始化状态（适用于新的 seq_len=32）。")

                # 现在使用 strict=False 加载“清理过”的权重字典
                # 这将加载所有匹配的层 (q,k,v) 并忽略 fc 层
                self.spectra_att_full_model.load_state_dict(spectra_weights_dict, strict=False)
                print("SpectraAtt 权重加载完成 (strict=False)。")
                # --- ### 修改结束 ### ---
                
            except FileNotFoundError:
                print(f"警告: 未找到 SpectraAtt 权重文件 {spectra_weights_path}")
            except Exception as e:
                print(f"加载 SpectraAtt 权重时出错: {e}")


    def forward(self, x: torch.Tensor, return_features: bool = False) -> torch.Tensor:
        """
        前向传播。
        输入 x 期望形状为 (B, 1, F, T)，例如 (B, 1, 512, 512)
        """
        
        # --- 分支 A: Res_Inc_Net (骨干) ---
        res_feat_map = self.res_backbone(x) 
        res_feat = self.res_pool(res_feat_map).squeeze(-1)

        # --- 分支 B: 频谱注意力 (骨干) ---
        x_stem = self.spectra_stem(x)
        sp_map = self.spectra_att_block(x_stem)
        sp_feat = sp_map.mean(dim=-1)
        
        # --- 融合与分类 ---
        feat = torch.cat([res_feat, sp_feat], dim=-1)

        if return_features:
            return feat

        logits = self.classifier(feat)
        return logits

    def freeze_backbones(self):
        """
        辅助函数：冻结两个骨干网络的参数，以便只训练分类头。
        """
        print("冻结 ResNet 和 SpectraAtt 骨干网络...")
        
        for param in self.res_backbone.parameters():
            param.requires_grad = False
            
        for param in self.spectra_stem.parameters():
            param.requires_grad = False
        for param in self.spectra_att_block.parameters():
            param.requires_grad = False
            
    def backbones_eval(self):
        """
        辅助函数：将骨干网络设置为 .eval() 模式。
        """
        # print("将 ResNet 和 SpectraAtt 骨干网络设置为 .eval() 模式 (用于 BN 和 Dropout)...")
        self.res_backbone.eval()
        self.spectra_stem.eval()
        self.spectra_att_block.eval()


# =============================================================================
#  使用示例 (如果此文件被直接运行)
# =============================================================================
if False:
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"--- 正在 {device} 上测试 HybridRFClassifier ---")

    # 1. 定义超参数 (应与您的数据匹配)
    NUM_CLASSES = 10
    FREQ_BINS = 512  # F
    SEQ_LEN = 512    # T
    STEM_POOL = 16   # SpectraAtt 的下采样因子
    
    # --- 修改：定义权重路径 ---
    RES_WEIGHTS_PATH = '/home/kk/lzk/SRSpecNet/1110/SW_Res_Inc_Net_base_SENet/model.pth'
    SPECTRA_WEIGHTS_PATH = '/home/kk/lzk/SRSpecNet/1111/rf_att_medium_downsample_16/model.pth'

    # 2. 实例化混合模型
    # --- 修改：在初始化时传入路径 ---
    model = HybridRFClassifier(
        num_classes=NUM_CLASSES,
        spectra_feat_dim=FREQ_BINS,
        freq_bins=FREQ_BINS,
        seq_len=SEQ_LEN,
        stem_pool_factor=STEM_POOL,
        res_weights_path=RES_WEIGHTS_PATH,       # <-- 传入路径
        spectra_weights_path=SPECTRA_WEIGHTS_PATH, # <-- 传入路径
        device=device                            # <-- 传入设备
    ).to(device)

    # 3. (模拟) 加载您预训练的权重
    # --- 这部分已在 __init__ 中自动完成 ---
    print("\n--- 权重加载已在 __init__ 中尝试 ---")


    # 4. 冻结骨干网络 (用于迁移学习)
    model.freeze_backbones()
    
    # 5. 将骨干网络设置为 eval 模式 (!!!)
    model.backbones_eval()
    
    # 6. 将整个模型设置为 train 模式
    model.train() 

    # 7. 检查参数的可训练状态
    print("\n--- 检查可训练参数 ---")
    total_params = 0
    trainable_params = 0
    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            print(f"[可训练] {name} (shape: {param.shape})")
    
    print(f"\n总参数: {total_params / 1e6:.2f} M")
    print(f"可训练参数: {trainable_params / 1e3:.2f} K")
    print("----------------------------\n")

    # 8. 测试前向传播
    dummy_input = torch.randn(2, 1, FREQ_BINS, SEQ_LEN).to(device)
    print(f"输入形状: {dummy_input.shape}")
    
    logits = model(dummy_input)
    
    print(f"输出 Logits 形状: {logits.shape}")
    assert logits.shape == (2, NUM_CLASSES)

    # 9. (可选) 测试特征提取
    model.eval() # 现在将整个模型设置为 eval 模式
    features = model(dummy_input, return_features=True)
    print(f"输出特征形状: {features.shape}")
    assert features.shape == (2, model.fused_dim)
    
    print("\n--- 混合模型测试成功! ---")