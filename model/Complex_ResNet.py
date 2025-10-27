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

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):
        super(BasicBlock, self).__init__()
        self.conv1 = ComplexConv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=stride, padding=1)
        self.bn1 = ComplexBatchNorm2d(out_channel)
        self.relu = ComplexReLU()

        self.conv2 = ComplexConv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1)
        self.bn2 = ComplexBatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self, x):
        # x应该是(real, imag)元组
        xr, xi = x
        identity = x
        
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(xr, xi)
        out = self.bn1(*out)
        out = self.relu(*out)

        out = self.conv2(*out)
        out = self.bn2(*out)

        # 复值残差连接
        identity_r, identity_i = identity
        out_r, out_i = out
        out = (out_r + identity_r, out_i + identity_i)
        out = self.relu(*out)

        return out

class ResNet(nn.Module):

    def __init__(self,
                 block,
                 blocks_num,
                 num_classes=1000,
                 include_top=True,
                 groups=1,
                 width_per_group=64):
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64

        self.groups = groups
        self.width_per_group = width_per_group

        self.conv1 = ComplexConv2d(1, self.in_channel, kernel_size=7, stride=2, padding=3)
        self.bn1 = ComplexBatchNorm2d(self.in_channel)

        self.relu = ComplexReLU(inplace=True)
        self.maxpool = ComplexMaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
        if self.include_top:
            self.avgpool = ComplexAdaptiveAvgPool2d((1, 1))  # 自适应全局平均池化
            self.fc = ComplexLinear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

        self.thresholds = nn.Parameter(torch.ones(num_classes) * 0.5)

    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            # 创建复值下采样序列
            class ComplexSequential(nn.Module):
                def __init__(self, *args):
                    super().__init__()
                    self.layers = nn.ModuleList(args)
                
                def forward(self, x):
                    xr, xi = x
                    for layer in self.layers:
                        xr, xi = layer(xr, xi)
                    return xr, xi
            
            downsample = ComplexSequential(
                ComplexConv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride),
                ComplexBatchNorm2d(channel * block.expansion)
            )

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

        # 创建复值序列容器
        class ComplexSequentialLayers(nn.Module):
            def __init__(self, layers):
                super().__init__()
                self.layers = nn.ModuleList(layers)
            
            def forward(self, x):
                for layer in self.layers:
                    x = layer(x)
                return x

        return ComplexSequentialLayers(layers)

    def forward(self, x, return_features=False):
        # 输入x应该是复值数据，分离实部和虚部
        # if torch.is_complex(x):
        #     xr = x.real
        #     xi = x.imag
        # else:
        #     # 如果输入是实数，虚部设为0
        #     xr = x
        #     xi = torch.zeros_like(x)
        xr = x[:,:,:,:,0]
        xi = x[:,:,:,:,1]

        xr, xi = self.conv1(xr, xi)
        xr, xi = self.bn1(xr, xi)
        xr, xi = self.relu(xr, xi)
        xr, xi = self.maxpool(xr, xi)

        xr, xi = self.layer1((xr, xi))
        xr, xi = self.layer2((xr, xi))
        xr, xi = self.layer3((xr, xi))
        xr, xi = self.layer4((xr, xi))

        if return_features:
            # 返回复值特征的模长
            magnitude = torch.sqrt(xr**2 + xi**2)
            return magnitude
        
        if self.include_top:
            xr, xi = self.avgpool(xr, xi)
            xr = torch.flatten(xr, 1)
            xi = torch.flatten(xi, 1)
            xr, xi = self.fc(xr, xi)
            
            # 对于分类任务，通常返回复值的模长
            output = torch.sqrt(xr**2 + xi**2)
            return output

        return xr, xi


def resnet18(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet18-5c106cde.pth
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, include_top=include_top)

def resnet34(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet34-333f7ec4.pth
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)

# if __name__ == "__main__":
#     from torchsummary import summary
#     from thop import profile
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#     model = resnet18(num_classes=4, include_top=True)
#     model = model.to(device)

#     # summary(model, input_size=(1, 512))
#     inputs = torch.randn(1, 1, 512, 512).to(device)
#     flops, params = profile(model, (inputs,))
#     print('flops: ', flops, 'params: ', params)

#     # output = model(inputs)
#     # print('Output shape:', output.shape)