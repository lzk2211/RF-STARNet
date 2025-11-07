import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import math

class ESEModule(nn.Module):
    """
    实现论文中图2所示的增强型Squeeze-and-Excitation (ESE) 模块。
   
    """
    def __init__(self, channels, reduction=4):
        super(ESEModule, self).__init__()
        # 论文中提到压缩因子为4
        reduced_dim = channels // reduction
        
        # 对应图2中的 Avgpool
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        # 对应图2中的 F1 和 F2 (两个全连接层)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, reduced_dim, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_dim, channels, 1, bias=False)
        )
        
        # 对应图2中的 Sigmoid
        self.sigmoid = nn.Sigmoid()

    def _delta_activation(self, x):
        """
        实现论文中公式(11) 相关的delta激活函数。
        论文中定义为 delta(x) = max(0, min(1, 0.2*x + 0.5))
        这等价于一个手动的 HardSigmoid 或 clamp 操作。
        """
        return torch.clamp(0.2 * x + 0.5, 0, 1)

    def forward(self, x):
        """
        x 对应 H_dw^(j)
        """
        # h_dw^(j)
        h_dw = self.avgpool(x)
        
        # z^(j)
        z = self.fc(h_dw)
        
        # alpha^(j)
        alpha = self.sigmoid(x)
        
        # 对应公式 (11): s^(j) = delta(z - h_dw * alpha)
        # z 和 h_dw 需要广播到 x 的维度
        s_input = z - h_dw * alpha
        s = self._delta_activation(s_input)
        
        # 对应公式 (12): H_out^(j) = H_dw^(j) * s^(j)
        return x * s

class ETFClassifier(nn.Module):
    """
    实现论文中第III-C节描述的固定ETF分类器。
    权重是根据公式 (17) 生成的，并且在训练期间保持固定。
    """
    def __init__(self, feature_dim, num_classes):
        super(ETFClassifier, self).__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        # 先设置设备，后续创建权重时使用
        
        # 论文中设置 eta = 1
        self.eta = 1.0
        
        # 生成固定的ETF权重
        weight = self._create_etf_weights()
        
        # 将权重注册为buffer，使其成为模型的一部分但不可训练
        self.register_buffer('weight', weight) # 权重 C [N x S]

    def _create_etf_weights(self):
        """
        根据公式 (17) 生成ETF权重矩阵 C
        C = eta * sqrt(S / (S-1)) * Q * (I - (1/S) * 1)
        """
        S = self.num_classes
        N = self.feature_dim
        
        # 1. 创建一个随机矩阵并正交化，得到 Q (N x S)
        # Q^T * Q = I_S
        Q, _ = torch.linalg.qr(torch.randn(N, S))
        
        # 2. 创建 M = (I - (1/S) * 1) (S x S)
        I = torch.eye(S)
        one_s = torch.ones(S, S)
        M = I - (1.0 / S) * one_s
        
        # 3. 计算 C = eta * sqrt(S / (S-1)) * Q * M
        scale = float(self.eta) * ( (self.num_classes / (self.num_classes - 1.0)) ** 0.5 )
        C = (Q @ M) * scale
        
        # C 的形状为 [feature_dim, num_classes]
        return C

    def forward(self, x):
        """
        x 是输入的特征向量 (x_bar)
        x 的形状: [batch_size, feature_dim]
        
        根据公式 (18) 和 (299) 计算logits
        logits = C^T * x
        
        self.weight 是 C (N x S)
        self.weight.T 是 C^T (S x N)
        """
        # F.linear(x, W) 计算 x * W^T
        # 我们需要 x * C，即 x * self.weight
        # 或者 F.linear(x, self.weight.T)
        logits = F.linear(x, self.weight.T)
        return logits

class MobileNetV3Small_ETF(nn.Module):
    """
    组装 MobileNetV3-Small-ETF 模型。
    
    注意：此实现仅替换了分类器。
    完整的论文实现 还需要在骨干网络内部
    将 SE 模块替换为 ESEModule。
    """
    def __init__(self, num_classes, pretrained_backbone=False):
        super(MobileNetV3Small_ETF, self).__init__()
        
        # 加载 MobileNetV3-Small 骨干网络
        if pretrained_backbone:
            weights = models.MobileNet_V3_Small_Weights.DEFAULT
        else:
            weights = None
            
        backbone = models.mobilenet_v3_small(weights=weights)
        
        # 提取特征提取器和平均池化层
        self.features = backbone.features
        self.avgpool = backbone.avgpool
        
        # 获取论文中提到的特征维度 (N_pw^(11) = 576)
        feature_dim = backbone.classifier[0].in_features # 576
        
        # 替换为固定的ETF分类器
        self.classifier = ETFClassifier(feature_dim, num_classes)

    def forward(self, x):
        # 输入通道适配：若输入为单通道，则复制到3通道以匹配骨干
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        # 骨干网络
        x = self.features(x)
        x = self.avgpool(x)
        
        # 展平特征
        features = torch.flatten(x, 1) # 对应 x_bar
        
        # ETF 分类器
        logits = self.classifier(features)
        
        # 返回 logits 和 features，因为 features 需要用于 L_ETF 损失
        return logits, features
    
if False:
	from torchsummary import summary
	from thop import profile
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	model = MobileNetV3Small_ETF(num_classes=10)
	model = model.to(device)

	# summary(model, input_size=(1, 512))
	inputs = torch.randn(1, 1, 512, 512).to(device)
	flops, params = profile(model, (inputs,))
	print('flops: ', flops, 'params: ', params)
