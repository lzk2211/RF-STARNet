import torch.nn as nn
import torch

class BasicBlock1D(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, groups=1, **kwargs):
        super(BasicBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False, groups=groups)
        self.bn1 = nn.BatchNorm1d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
        self.bn2 = nn.BatchNorm1d(out_channel)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


class ResNet1D(nn.Module):

    def __init__(self,
                 block,
                 blocks_num,
                 num_classes=1000,
                 include_top=True,
                 groups=1,
                 width_per_group=64):
        super(ResNet1D, self).__init__()
        self.include_top = include_top
        self.in_channel = 64
        self.groups = groups
        self.width_per_group = width_per_group

        self.conv1 = nn.Conv1d(1, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
        self.dropout = nn.Dropout(p=0.5)
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool1d(1)  # output size = 1
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channel, channel * block.expansion, kernel_size=1,
                          stride=stride, bias=False, groups=self.groups),
                nn.BatchNorm1d(channel * block.expansion))

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

        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.mean(dim=-1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

        return x
    
    def get_features(self, x, layer=None):
        if layer is None:
            return self.forward(x)
        else:
            out = {}
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            if 'layer1' in layer:
                out['layer1'] = self.layer1(x)
            x = out['layer1'] if 'layer1' in layer else x

            if 'layer2' in layer:
                out['layer2'] = self.layer2(x)
            x = out['layer2'] if 'layer2' in layer else x

            if 'layer3' in layer:
                out['layer3'] = self.layer3(x)
            x = out['layer3'] if 'layer3' in layer else x

            if 'layer4' in layer:
                out['layer4'] = self.layer4(x)
            x = out['layer4'] if 'layer4' in layer else x

            return x


def resnet18_vector_group(num_classes=1000, include_top=True):
    return ResNet1D(BasicBlock1D, [2, 2, 2, 2], num_classes=num_classes, include_top=include_top, groups=32, width_per_group=4)

def resnet10_vector_group(num_classes=1000, include_top=True):
    return ResNet1D(BasicBlock1D, [1, 1, 1, 1], num_classes=num_classes, include_top=include_top, groups=32, width_per_group=4)

# from thop import profile
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# model = resnet18_vector_group(num_classes=10, include_top=True)
# model = model.to(device)
# inputs = torch.randn(1, 1, 512, 512).to(device) # 1D 频谱向量
# flops, params = profile(model, (inputs,), verbose=False)
# print('flops: ', flops, 'params: ', params)