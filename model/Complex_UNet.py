import torch
from torch import nn


class ComplexConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super().__init__()
        self.real_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.imag_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)

    def forward(self, xr, xi):
        real = self.real_conv(xr) - self.imag_conv(xi)
        imag = self.real_conv(xi) + self.imag_conv(xr)
        return real, imag


class ComplexBatchNorm2d(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.bn_real = nn.BatchNorm2d(num_features)
        self.bn_imag = nn.BatchNorm2d(num_features)

    def forward(self, xr, xi):
        return self.bn_real(xr), self.bn_imag(xi)


class ComplexReLU(nn.Module):
    def __init__(self, inplace=False):
        super().__init__()
        self.prelu_real = nn.ReLU(inplace=inplace)
        self.prelu_imag = nn.ReLU(inplace=inplace)

    def forward(self, xr, xi):
        return self.prelu_real(xr), self.prelu_imag(xi)

class ComplexMaxPool2d(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.maxpool_real = nn.MaxPool2d(kernel_size, stride, padding)
        self.maxpool_imag = nn.MaxPool2d(kernel_size, stride, padding)

    def forward(self, xr, xi):
        real = self.maxpool_real(xr)
        imag = self.maxpool_imag(xi)
        return real, imag

class ComplexAvgPool2d(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.avgpool_real = nn.AvgPool2d(kernel_size, stride, padding)
        self.avgpool_imag = nn.AvgPool2d(kernel_size, stride, padding)

    def forward(self, xr, xi):
        real = self.avgpool_real(xr)
        imag = self.avgpool_imag(xi)
        return real, imag
    
class ComplexAdaptiveAvgPool2d(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.adaptive_avgpool_real = nn.AdaptiveAvgPool2d(output_size)
        self.adaptive_avgpool_imag = nn.AdaptiveAvgPool2d(output_size)

    def forward(self, xr, xi):
        real = self.adaptive_avgpool_real(xr)
        imag = self.adaptive_avgpool_imag(xi)
        return real, imag
    
class ComplexLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear_real = nn.Linear(in_features, out_features)
        self.linear_imag = nn.Linear(in_features, out_features)

    def forward(self, xr, xi):
        real = self.linear_real(xr) - self.linear_imag(xi)
        imag = self.linear_real(xi) + self.linear_imag(xr)
        return real, imag

class ComplexConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, bias=False):
        super().__init__()
        self.real_conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding, bias=bias)
        self.imag_conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding, bias=bias)

    def forward(self, xr, xi):
        real = self.real_conv_transpose(xr) - self.imag_conv_transpose(xi)
        imag = self.real_conv_transpose(xi) + self.imag_conv_transpose(xr)
        return real, imag

class DoubleConv(nn.Module):
    """双卷积层 (conv -> BN -> ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = ComplexConv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = ComplexBatchNorm2d(out_channels)
        self.relu1 = ComplexReLU()
        
        self.conv2 = ComplexConv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = ComplexBatchNorm2d(out_channels)
        self.relu2 = ComplexReLU()

    def forward(self, x):
        xr, xi = x
        xr, xi = self.conv1(xr, xi)
        xr, xi = self.bn1(xr, xi)
        xr, xi = self.relu1(xr, xi)
        
        xr, xi = self.conv2(xr, xi)
        xr, xi = self.bn2(xr, xi)
        xr, xi = self.relu2(xr, xi)
        
        return xr, xi

class Down(nn.Module):
    """下采样层 (maxpool -> double conv)"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.ModuleList([
            ComplexMaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        ])

    def forward(self, x):
        xr, xi = x
        xr, xi = self.maxpool_conv[0](xr, xi)
        xr, xi = self.maxpool_conv[1]((xr, xi))
        return xr, xi

class Up(nn.Module):
    """上采样层 (upconv -> concat -> double conv)"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        if bilinear:
            # 使用双线性插值 + 1x1卷积
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            # 使用转置卷积
            self.up = ComplexConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

        self.bilinear = bilinear

    def forward(self, x1, x2):
        """
        x1: 来自decoder的特征
        x2: 来自encoder的skip connection特征
        """
        x1r, x1i = x1
        x2r, x2i = x2
        
        if self.bilinear:
            x1r = self.up(x1r)
            x1i = self.up(x1i)
        else:
            x1r, x1i = self.up(x1r, x1i)

        # 确保尺寸匹配
        diffY = x2r.size()[2] - x1r.size()[2]
        diffX = x2r.size()[3] - x1r.size()[3]

        x1r = nn.functional.pad(x1r, [diffX // 2, diffX - diffX // 2,
                                      diffY // 2, diffY - diffY // 2])
        x1i = nn.functional.pad(x1i, [diffX // 2, diffX - diffX // 2,
                                      diffY // 2, diffY - diffY // 2])

        # 拼接特征
        xr = torch.cat([x2r, x1r], dim=1)
        xi = torch.cat([x2i, x1i], dim=1)
        
        return self.conv((xr, xi))

class ComplexUNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=1000, bilinear=True, include_top=True):
        super(ComplexUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.include_top = include_top

        # 编码器
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)

        # 线性层（可选）
        if self.include_top:
            self.avgpool = ComplexAdaptiveAvgPool2d((1, 1))
            self.fc = ComplexLinear(1024 // factor, n_classes)

        # 解码器
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = ComplexConv2d(64, 1, kernel_size=1)

        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

        self.thresholds = nn.Parameter(torch.ones(n_classes) * 0.5)

    def forward(self, x, return_features=False):
        # 输入处理
        xr = x[:,:,:,:,0]
        xi = x[:,:,:,:,1]

        # 编码器
        x1r, x1i = self.inc((xr, xi))
        x2r, x2i = self.down1((x1r, x1i))
        x3r, x3i = self.down2((x2r, x2i))
        x4r, x4i = self.down3((x3r, x3i))
        x5r, x5i = self.down4((x4r, x4i))

        # 线性层处理（如果启用）
        if self.include_top:
            # 全局平均池化 + 线性层
            pooled_r, pooled_i = self.avgpool(x5r, x5i)
            pooled_r = torch.flatten(pooled_r, 1)
            pooled_i = torch.flatten(pooled_i, 1)
            fc_r, fc_i = self.fc(pooled_r, pooled_i)
            
            # # 将线性层输出重新reshape为特征图
            # # 假设原始特征图大小为 H x W
            # H, W = x5r.shape[2], x5r.shape[3]
            # fc_r = fc_r.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
            # fc_i = fc_i.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
            
            # # 将线性层输出与瓶颈层特征相加或拼接
            # x5r = x5r + fc_r
            # x5i = x5i + fc_i
            return torch.sqrt(fc_r**2 + fc_i**2)

        if return_features:
            # 返回瓶颈层特征的模长
            magnitude = torch.sqrt(x5r**2 + x5i**2)
            return magnitude

        # 解码器
        xr, xi = self.up1((x5r, x5i), (x4r, x4i))
        xr, xi = self.up2((xr, xi), (x3r, x3i))
        xr, xi = self.up3((xr, xi), (x2r, x2i))
        xr, xi = self.up4((xr, xi), (x1r, x1i))
        xr, xi = self.outc(xr, xi)

        y = torch.concat((xr.unsqueeze(-1), xi.unsqueeze(-1)), dim=-1)
        return y


def complex_unet(n_channels=1, n_classes=1000, bilinear=True, include_top=True):
    """创建复值UNet模型"""
    return ComplexUNet(n_channels=n_channels, n_classes=n_classes, bilinear=bilinear, include_top=include_top)


# if __name__ == "__main__":
#     from torchsummary import summary
#     from thop import profile
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#     model = complex_unet(n_classes=4, include_top=False)
#     model = model.to(device)

#     # summary(model, input_size=(1, 512))
#     inputs = torch.randn(1, 1, 512, 512, 2).to(device)
#     flops, params = profile(model, (inputs,))
#     print('flops: ', flops, 'params: ', params)

#     output = model(inputs)
#     print('Output shape:', output.shape)
