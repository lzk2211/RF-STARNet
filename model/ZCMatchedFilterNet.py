import torch
import torch.nn as nn
import torch.nn.functional as F

class ZCMatchedFilterNet(nn.Module):
    def __init__(self, classnum, init_filter_size=(7, 7)):
        """
        初始化模型
        :param classnum: 你的类别数量
        :param init_filter_size: 初始“匹配滤波器”的尺寸 (F, T)。
                                 你需要调整这个尺寸，使其大致匹配ZC序列在频谱图上的“足迹”。
                                 (7, 7) 只是一个起始点。
        """
        super(ZCMatchedFilterNet, self).__init__()
        self.classnum = classnum
        
        # -----------------------------------------------------------------
        # 1. 可学习的“匹配滤波器”层 (完全按照你的设想)
        # -----------------------------------------------------------------
        # kernel_size 需要实验。如果ZC序列在F轴上很宽，T轴上很窄，
        # 你可以尝试 (15, 5) 或 (21, 3) 等。
        # 我们使用 padding 来保持尺寸不变，以便观察。
        padding_F = init_filter_size[0] // 2
        padding_T = init_filter_size[1] // 2
        
        self.matched_filter_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=1, 
                out_channels=classnum, 
                kernel_size=init_filter_size, 
                padding=(padding_F, padding_T),
                bias=False # 纯粹的“相关”操作可以不加偏置
            ),
            nn.BatchNorm2d(classnum), # 标准化响应强度
            nn.ReLU(inplace=True)
            # 输出: (bs, classnum, 512, 512)
        )
        
        # -----------------------------------------------------------------
        # 2. 特征提取主干网络
        # -----------------------------------------------------------------
        # 这里的输入通道是 classnum，它将学习“匹配响应”的组合
        self.backbone = nn.Sequential(
            # Block 1
            nn.Conv2d(classnum, 32, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2), # 尺寸 -> 256x256
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2), # 尺寸 -> 128x128
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2), # 尺寸 -> 64x64
            
            # Block 4
            nn.Conv2d(128, 256, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2)  # 尺寸 -> 32x32
        )
        
        # -----------------------------------------------------------------
        # 3. 分类头 (FC层)
        # -----------------------------------------------------------------
        # 使用自适应平均池化，将 (bs, 256, 32, 32) 变为 (bs, 256, 1, 1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 展平
        self.flatten = nn.Flatten()
        
        # 最后的FC层
        self.classifier = nn.Linear(256, classnum)

    def forward(self, x):
        # x 初始维度: (bs, 1, 512, 512)
        
        # 1. 通过“匹配滤波器”层
        # h1 维度: (bs, classnum, 512, 512)
        h1 = self.matched_filter_layer(x) 
        
        # 2. 通过特征提取主干
        # h2 维度: (bs, 256, 32, 32)
        h2 = self.backbone(h1)
        
        # 3. 通过分类头
        # h3 维度: (bs, 256, 1, 1)
        h3 = self.avgpool(h2)
        
        # h4 维度: (bs, 256)
        h4 = self.flatten(h3)
        
        # out 维度: (bs, classnum)
        out = self.classifier(h4)
        
        return out

# from thop import profile
# # 假设你有 10 个类别
# CLASS_NUM = 25
# # 假设你的输入批次
# # (bs, 1, 512, 512)
# dummy_input = torch.randn(16, 1, 512, 512) 

# # -------------------------------------------------------
# # 1. 实例化模型
# # -------------------------------------------------------
# # 你可以试验不同的滤波器尺寸，比如 (11, 11) 或 (9, 5)
# model = ZCMatchedFilterNet(classnum=CLASS_NUM, init_filter_size=(7, 7))

# # -------------------------------------------------------
# # 2. 前向传播
# # -------------------------------------------------------
# output = model(dummy_input)

# # -------------------------------------------------------
# # 3. 打印输出
# # -------------------------------------------------------
# print(f"输入维度: {dummy_input.shape}")
# print(f"输出维度: {output.shape}") 
# # 期望输出: torch.Size([16, 10])

# # -------------------------------------------------------
# # 4. 检查模型参数量
# # -------------------------------------------------------
# total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# print(f"模型总参数量: {total_params:,}")

# inputs = torch.randn(1, 1, 512, 512)
# flops, params = profile(model, (inputs,))

# print(f"输入形状: {inputs.shape}")
# print(f'FLOPs: {flops / 1e9:.2f} G')
# print(f'Params: {params / 1e6:.2f} M')