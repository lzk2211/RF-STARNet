import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class EfficientNetB0(nn.Module):
	"""
	EfficientNet-B0 封装：
	- 复用 torchvision 的 efficientnet_b0 作为骨干
	- 保留自适应池化并替换分类器为线性层
	- 兼容单通道输入：会复制为 3 通道
	- forward 返回 (logits, features)，以适配当前训练脚本

	属性：
	- features: 特征提取器
	- avgpool: 自适应池化
	- classifier: 线性分类器 (其权重会被 CombinedLoss 使用)
	"""

	def __init__(self, num_classes: int, pretrained_backbone: bool = False):
		super().__init__()

		weights = (
			models.EfficientNet_B0_Weights.DEFAULT if pretrained_backbone else None
		)
		backbone = models.efficientnet_b0(weights=weights)

		# 提取骨干结构
		self.features = backbone.features
		self.avgpool = backbone.avgpool  # AdaptiveAvgPool2d(1)

		# 获取特征维度（原模型 classifier[1] 为 Linear）
		feature_dim = backbone.classifier[1].in_features

		# 替换分类器为简单线性层，保持属性名为 classifier，便于现有损失读取权重
		self.classifier = nn.Linear(feature_dim, num_classes)

	def forward(self, x: torch.Tensor):
		# 输入通道适配：若是单通道，则复制到 3 通道
		if x.shape[1] == 1:
			x = x.repeat(1, 3, 1, 1)

		x = self.features(x)
		x = self.avgpool(x)
		features = torch.flatten(x, 1)
		logits = self.classifier(features)
		return logits


if False:
	# 可选的快速自检与 FLOPs 估算（需要 thop）
	try:
		from thop import profile
	except Exception:
		profile = None

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	model = EfficientNetB0(num_classes=10, pretrained_backbone=False).to(device)

	# 假设输入为 (B, 1, 512, 512)
	inputs = torch.randn(1, 1, 512, 512).to(device)
	logits = model(inputs)
	print("logits:", logits.shape)

	if profile is not None:
		flops, params = profile(model, (inputs,))
		print("FLOPs:", flops, "Params:", params)
