import torch
import torch.nn as nn
from typing import Optional

try:
	from model1119.F_cnn import *
	from model1119.T_att import *
except ImportError as e:
	print("未能在 'model1119' 文件夹下找到模块，尝试从当前目录导入...")
	from F_cnn import *
	from T_att import *

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
		mode: str = 'Train'
	) -> None:
		super().__init__()
		self.mode = mode

		if use_resinc_variant == 'large':
			F_cnn = F_cnn_base
			T_att = T_att_base
			C_res = 512
		elif use_resinc_variant == 'base':
			F_cnn = F_cnn_base
			T_att = T_att_base
			C_res = 512
		elif use_resinc_variant == 'small':
			F_cnn = F_cnn_tiny
			T_att = T_att_tiny
			C_res = 256
		elif use_resinc_variant == 'tiny':
			F_cnn = F_cnn_tiny
			T_att = T_att_tiny
			C_res = 256
		elif use_resinc_variant == 'mini':
			F_cnn = F_cnn_mini
			T_att = T_att_mini
			C_res = 128  # 与 Res_Inc_Net_mini 的默认配置一致
		elif use_resinc_variant == 'nano':
			F_cnn = F_cnn_nano
			T_att = T_att_nano
			C_res = 64   # 与 Res_Inc_Net_nano 的默认配置一致

		# 分支A: 1D ResNet 骨干（不含顶层分类头）
		# include_top=False 时，forward 返回 layer4 的特征图 (B, C_res, L_res)
		self.res_backbone = F_cnn(num_classes=num_classes, include_top=False)
		self.res_pool = nn.AdaptiveAvgPool1d(1)  # (B, C_res, L_res) -> (B, C_res, 1)

		# 分支B: 频谱注意力 (直接使用注意力层而非分类器，便于抽取特征)
		self.spectra_att = T_att(
			freq_bins=512, 
			seq_len=512, 
			num_classes=num_classes, 
			include_top=False # <-- 测试 False
		)

		# 维度推断：
		# - Res 分支通道数：对于 base(2,2,2,2)+BasicBlock，最终 C_res 通常为 512
		# - Spectra 分支经时间平均后维度为 freq_bins
		C_sp = 512

		feat_dim = C_res + freq_bins

		# 轻量分类头：参数显著减少（LN + Dropout + Linear）
		# 原结构为两层全连接（feat_dim->hidden_dim->num_classes），参数较多；
		# 现改为单层线性头，通常已足够应对拼接后的高维特征。
		self.classifier = nn.Sequential(
			nn.LayerNorm(feat_dim),
			nn.Dropout(dropout),
			nn.Linear(feat_dim, num_classes),
		)

		reduction_ratio = 16 # 可以调整 (8, 16, 32)
		bottleneck_dim = max(16, C_res // reduction_ratio)

		self.gate_projector = nn.Sequential(
			nn.Linear(C_res, bottleneck_dim),
			nn.ReLU(inplace=True),
			nn.Linear(bottleneck_dim, C_sp)
			# Sigmoid 在 forward 中调用
		)

	def forward(self, x: torch.Tensor, return_features: bool = False) -> torch.Tensor:
		# ... (res_feat 和 sp_feat 的计算不变)

		res_feat = self.res_pool(self.res_backbone(x)).squeeze(-1) # (B, 512)

		sp_feat = self.spectra_att(x).mean(dim=-1)                 # (B, 512)		
		# 方案二：低秩适配器 (加法融合)
		# (self.gate_projector 在 __init__ 中定义为方案一的瓶颈)
		gate_signal = self.gate_projector(res_feat)
		gate_weights = torch.sigmoid(gate_signal)
		gated_sp_feat = sp_feat * gate_weights

		# --- 关键修改：在拼接前进行 L2 归一化 ---
		# 确保两个特征在数值上是平等的，防止量级消失
		# res_feat = F.normalize(res_feat, p=2, dim=-1)
		# sp_feat = F.normalize(sp_feat, p=2, dim=-1)

		# 融合
		feat = torch.cat([res_feat, gated_sp_feat], dim=-1)  # (B, C_res + F)
		if return_features:
			return feat
		logits = self.classifier(feat)

		return logits		

def F_T(num_classes: int = 25,
				 freq_bins: int = 512,
				 seq_len: int = 128,
				 groups: int = 8,
				 att_dim: int = 32,
				 k_len: int = 64,
				 dropout: float = 0.1,
				 hidden_dim: Optional[int] = 512,
				 use_resinc_variant: str = "base"):
	"""工厂函数，便于在 switch_model 中注册调用。"""
	return ResInc_SpAtt_Classifier(
		num_classes=num_classes,
		use_resinc_variant=use_resinc_variant,
		freq_bins=freq_bins,
		seq_len=seq_len,
		groups=groups,
		att_dim=att_dim,
		k_len=k_len,
		dropout=dropout,
		hidden_dim=hidden_dim,
	)

if __name__ == "__main__":
	# from torchsummary import summary
	from thop import profile
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	for name in ['nano', 'mini', 'tiny', 'small', 'base', 'large']:
		print(f"--- Testing {name} ---")
		model = F_T(num_classes=10, use_resinc_variant=name)
		model = model.to(device)
		inputs = torch.randn(1, 1, 512, 512).to(device) # 1D 频谱向量
		flops, params = profile(model, (inputs,), verbose=False)
		print(name, 'flops: ', flops, 'params: ', params)
