import torch
import torch.nn as nn
from typing import Optional

# 注意：Res_Inc_Net.py 在导入时包含FLOPs测试代码；该仓库已在其他位置导入过它。
# 这里仍按常规方式导入其工厂函数与结构，供骨干网络使用。
from model.Res_Inc_Net import Res_Inc_Net_base
from model.SpectraAtt_noProjected import GroupedProjectedAttentionRF


class ResInc_SpAtt_Classifier(nn.Module):
	"""
	联合模型：将 1D ResNet(Res_Inc_Net) 与 频谱注意力(GroupedProjectedAttentionRF) 的特征拼接后分类。

	输入:  x (B, 1, F, T)，默认 (B, 1, 512, 512)
	流程:
	  - 分支A: Res_Inc_Net_base(include_top=False) 接收 x，内部会对最后一维做 mean 得到 1D 序列，
			   输出特征图 (B, C_res, L_res)，随后自适应池化为 (B, C_res)
	  - 分支B: GroupedProjectedAttentionRF 接收 x，输出 (B, F, T)，随后对时间维平均为 (B, F)
	  - 拼接: [feat_res, feat_sp] -> (B, C_res + F)
	  - 分类: MLP 头得到 logits (B, num_classes)

	额外:
	  - return_features=True 时返回拼接后的特征向量，便于 t-SNE 可视化。
	"""

	def __init__(
		self,
		num_classes: int = 25,
		# Res_Inc_Net 分支配置
		use_resinc_variant: str = "base",  # 预留选项，当前仅使用 base
		# SpectraAtt 分支配置
		freq_bins: int = 512,
		seq_len: int = 128,
		groups: int = 8,
		att_dim: int = 32,
		k_len: int = 64,
		dropout: float = 0.1,
		# 头部配置
		hidden_dim: Optional[int] = 512,
	) -> None:
		super().__init__()

		# 分支A: 1D ResNet 骨干（不含顶层分类头）
		# include_top=False 时，forward 返回 layer4 的特征图 (B, C_res, L_res)
		assert use_resinc_variant == "base", "当前实现使用 Res_Inc_Net_base，可按需扩展"
		self.res_backbone = Res_Inc_Net_base(num_classes=num_classes, include_top=False)
		self.res_pool = nn.AdaptiveAvgPool1d(1)  # (B, C_res, L_res) -> (B, C_res, 1)

		# 分支B: 频谱注意力 (直接使用注意力层而非分类器，便于抽取特征)
		self.spectra_att = GroupedProjectedAttentionRF(
			freq_bins=freq_bins,
			att_dim=att_dim,
			groups=groups,
			seq_len=seq_len,
			k_len=k_len,
			use_pos=True,
			dropout=dropout,
		)

		# 维度推断：
		# - Res 分支通道数：对于 base(2,2,2,2)+BasicBlock，最终 C_res 通常为 512
		# - Spectra 分支经时间平均后维度为 freq_bins
		C_res = 512  # 与 Res_Inc_Net_base 的默认配置一致
		C_sp = freq_bins

		feat_dim = C_res + C_sp
		if hidden_dim is None:
			hidden_dim = max(256, feat_dim // 2)

		self.classifier = nn.Sequential(
			nn.LayerNorm(feat_dim),
			nn.Dropout(dropout),
			nn.Linear(feat_dim, hidden_dim),
			nn.ReLU(inplace=True),
			nn.Dropout(dropout),
			nn.Linear(hidden_dim, num_classes),
		)
       
        # 新增一个门控线性层
		self.fusion_gate = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.Sigmoid()
        )

        # 头部 MLP
		self.classifier = nn.Sequential(
            nn.LayerNorm(feat_dim),
            # ... (后续代码不变)
        )

	def forward(self, x: torch.Tensor, return_features: bool = False) -> torch.Tensor:
		"""
		x: (B, 1, F, T)
		return:
		  - 若 return_features=True: (B, C_res + F)
		  - 否则: logits (B, num_classes)
		"""
		# 分支A: Res_Inc_Net
		# 注意: Res_Inc_Net 内部会先对最后一维做 mean -> (B, 1, F)
		res_feat_map = self.res_backbone(x)             # include_top=False -> (B, C_res, L_res)
		res_feat = self.res_pool(res_feat_map).squeeze(-1)  # (B, C_res)

		# 分支B: 频谱注意力
		sp_map = self.spectra_att(x)  # (B, F, T)
		sp_feat = sp_map.mean(dim=-1)  # (B, F)

		# --- 修改从这里开始 ---
        # 原始拼接
		combined_feat = torch.cat([res_feat, sp_feat], dim=-1)		
		# 计算门控权重
		gate = self.fusion_gate(combined_feat)

		# 加权融合
		# 门控权重分别作用于 res 特征和 sp 特征部分
		gated_res_feat = gate[:, :res_feat.shape[1]] * res_feat
		gated_sp_feat = gate[:, res_feat.shape[1]:] * sp_feat		
		# 获得最终融合特征
		feat = torch.cat([gated_res_feat, gated_sp_feat], dim=-1)

		if return_features:
			return feat

		logits = self.classifier(feat)
		return logits


def ResInc_SpAtt(num_classes: int = 25,
				 freq_bins: int = 512,
				 seq_len: int = 128,
				 groups: int = 8,
				 att_dim: int = 32,
				 k_len: int = 64,
				 dropout: float = 0.1,
				 hidden_dim: Optional[int] = 512):
	"""工厂函数，便于在 switch_model 中注册调用。"""
	return ResInc_SpAtt_Classifier(
		num_classes=num_classes,
		use_resinc_variant="base",
		freq_bins=freq_bins,
		seq_len=seq_len,
		groups=groups,
		att_dim=att_dim,
		k_len=k_len,
		dropout=dropout,
		hidden_dim=hidden_dim,
	)

if False:
	from torchsummary import summary
	from thop import profile
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	model = ResInc_SpAtt(num_classes=10)
	model = model.to(device)

	# summary(model, input_size=(1, 512))
	inputs = torch.randn(1, 1, 512, 512).to(device)
	flops, params = profile(model, (inputs,))
	print('flops: ', flops, 'params: ', params)
