import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo

BatchNorm2d = nn.BatchNorm2d

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'deformable_resnet18', 'deformable_resnet50',
           'resnet152']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def constant_init(module, constant, bias=0):
    nn.init.constant_(module.weight, constant)
    if hasattr(module, 'bias'):
        nn.init.constant_(module.bias, bias)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dcn=None):
        super(BasicBlock, self).__init__()
        self.with_dcn = dcn is not None
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.with_modulated_dcn = False
        if not self.with_dcn:
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        else:
            from torchvision.ops import DeformConv2d
            deformable_groups = dcn.get('deformable_groups', 1)
            offset_channels = 18
            self.conv2_offset = nn.Conv2d(planes, deformable_groups * offset_channels, kernel_size=3, padding=1)
            self.conv2 = DeformConv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # out = self.conv2(out)
        if not self.with_dcn:
            out = self.conv2(out)
        else:
            offset = self.conv2_offset(out)
            out = self.conv2(out, offset)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, dcn=None):
        super(Bottleneck, self).__init__()
        self.with_dcn = dcn is not None
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes)
        self.with_modulated_dcn = False
        if not self.with_dcn:
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        else:
            deformable_groups = dcn.get('deformable_groups', 1)
            from torchvision.ops import DeformConv2d
            offset_channels = 18
            self.conv2_offset = nn.Conv2d(planes, deformable_groups * offset_channels, stride=stride, kernel_size=3, padding=1)
            self.conv2 = DeformConv2d(planes, planes, kernel_size=3, padding=1, stride=stride, bias=False)
        self.bn2 = BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dcn = dcn
        self.with_dcn = dcn is not None

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # out = self.conv2(out)
        if not self.with_dcn:
            out = self.conv2(out)
        else:
            offset = self.conv2_offset(out)
            out = self.conv2(out, offset)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


import math
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo


class ResNet(nn.Module):
    def __init__(self, block, layers, in_channels=3, dcn=None):
        super(ResNet, self).__init__()
        self.dcn = dcn
        self.inplanes = 64
        self.out_channels = []

        # 原始 ResNet 结构
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dcn=dcn)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dcn=dcn)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dcn=dcn)

        # ✅ 每层都加一个可学习的 `alpha`
        self.alpha2 = nn.Parameter(torch.ones(1))  # 用于 x2
        self.alpha3 = nn.Parameter(torch.ones(1))  # 用于 x3
        self.alpha4 = nn.Parameter(torch.ones(1))  # 用于 x4
        self.alpha5 = nn.Parameter(torch.ones(1))  # 用于 x5
        self.sigmoid = nn.Sigmoid()  # 让 alpha 在 (0,1) 之间

        # ✅ 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        if self.dcn is not None:
            for m in self.modules():
                if isinstance(m, Bottleneck) or isinstance(m, BasicBlock):
                    if hasattr(m, 'conv2_offset'):
                        constant_init(m.conv2_offset, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dcn=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample, dcn=dcn)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dcn=dcn))

        self.out_channels.append(planes * block.expansion)
        return nn.Sequential(*layers)

    def forward(self, snow_img, img):
        """双输入 + 各层特征加权融合"""
        # 处理雪图
        x_snow = self.conv1(snow_img)
        x_snow = self.bn1(x_snow)
        x_snow = self.relu(x_snow)
        x_snow = self.maxpool(x_snow)

        # 处理去雪图
        x_desnow = self.conv1(img)
        x_desnow = self.bn1(x_desnow)
        x_desnow = self.relu(x_desnow)
        x_desnow = self.maxpool(x_desnow)

        # ResNet 各层
        x2_snow = self.layer1(x_snow)
        x3_snow = self.layer2(x2_snow)
        x4_snow = self.layer3(x3_snow)
        x5_snow = self.layer4(x4_snow)

        x2_desnow = self.layer1(x_desnow)
        x3_desnow = self.layer2(x2_desnow)
        x4_desnow = self.layer3(x3_desnow)
        x5_desnow = self.layer4(x4_desnow)

        # ✅ 每层进行加权融合
        alpha2 = self.sigmoid(self.alpha2)
        alpha3 = self.sigmoid(self.alpha3)
        alpha4 = self.sigmoid(self.alpha4)
        alpha5 = self.sigmoid(self.alpha5)

        x2 = alpha2 * x2_snow + (1 - alpha2) * x2_desnow
        x3 = alpha3 * x3_snow + (1 - alpha3) * x3_desnow
        x4 = alpha4 * x4_snow + (1 - alpha4) * x4_desnow
        x5 = alpha5 * x5_snow + (1 - alpha5) * x5_desnow

        return x2, x3, x4, x5


# ✅ `resnet18`
def resnet18(pretrained=True, **kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        assert kwargs['in_channels'] == 3, 'in_channels 必须是 3 (预训练模型限制)'
        print('加载 ImageNet 预训练模型')
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']), strict=False)
    return model

# class ChannelAttention(nn.Module):
#     def __init__(self, in_planes, ratio=16):
#         super(ChannelAttention, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
#
#         self.fc = nn.Sequential(
#             nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
#             nn.ReLU(),
#             nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
#         )
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         avg_out = self.fc(self.avg_pool(x))
#         max_out = self.fc(self.max_pool(x))
#         return self.sigmoid(avg_out + max_out)
#
# class ResNet(nn.Module):
#     def __init__(self, block, layers, in_channels=3):
#         super(ResNet, self).__init__()
#         # self.dcn = dcn
#         self.inplanes = 64
#         self.out_channels = []
#
#         self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#
#         self.layer1 = self._make_layer(block, 64, layers[0])
#         self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
#         self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
#         self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
#
#         self.ca2 = ChannelAttention(64)
#         self.ca3 = ChannelAttention(128)
#         self.ca4 = ChannelAttention(256)
#         self.ca5 = ChannelAttention(512)
#
#         self.alpha2 = nn.Parameter(torch.ones(1))
#         self.alpha3 = nn.Parameter(torch.ones(1))
#         self.alpha4 = nn.Parameter(torch.ones(1))
#         self.alpha5 = nn.Parameter(torch.ones(1))
#         self.sigmoid = nn.Sigmoid()
#
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#
#     def _make_layer(self, block, planes, blocks, stride=1):
#         downsample = None
#         if stride != 1 or self.inplanes != planes * block.expansion:
#             downsample = nn.Sequential(
#                 nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(planes * block.expansion),
#             )
#
#         layers = [block(self.inplanes, planes, stride, downsample)]
#         self.inplanes = planes * block.expansion
#         for _ in range(1, blocks):
#             layers.append(block(self.inplanes, planes))
#
#         self.out_channels.append(planes * block.expansion)
#         return nn.Sequential(*layers)
#
#     def forward(self, snow_img, img):
#         x_snow = self.relu(self.bn1(self.conv1(snow_img)))
#         x_snow = self.maxpool(x_snow)
#
#         x_desnow = self.relu(self.bn1(self.conv1(img)))
#         x_desnow = self.maxpool(x_desnow)
#
#         x2_snow = self.layer1(x_snow)
#         x3_snow = self.layer2(x2_snow)
#         x4_snow = self.layer3(x3_snow)
#         x5_snow = self.layer4(x4_snow)
#
#         x2_desnow = self.layer1(x_desnow)
#         x3_desnow = self.layer2(x2_desnow)
#         x4_desnow = self.layer3(x3_desnow)
#         x5_desnow = self.layer4(x4_desnow)
#
#         alpha2 = self.sigmoid(self.alpha2) * self.ca2(x2_snow + x2_desnow)
#         alpha3 = self.sigmoid(self.alpha3) * self.ca3(x3_snow + x3_desnow)
#         alpha4 = self.sigmoid(self.alpha4) * self.ca4(x4_snow + x4_desnow)
#         alpha5 = self.sigmoid(self.alpha5) * self.ca5(x5_snow + x5_desnow)
#
#         x2 = alpha2 * x2_snow + (1 - alpha2) * x2_desnow
#         x3 = alpha3 * x3_snow + (1 - alpha3) * x3_desnow
#         x4 = alpha4 * x4_snow + (1 - alpha4) * x4_desnow
#         x5 = alpha5 * x5_snow + (1 - alpha5) * x5_desnow
#
#         return x2, x3, x4, x5
#
# def resnet18(pretrained=True, **kwargs):
#     model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
#     if pretrained:
#         assert kwargs['in_channels'] == 3, 'in_channels 必须是 3 (预训练模型限制)'
#         print('加载 ImageNet 预训练模型')
#         model.load_state_dict(model_zoo.load_url(model_urls['resnet18']), strict=False)
#     return model

def deformable_resnet18(pretrained=True, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], dcn=dict(deformable_groups=1), **kwargs)
    if pretrained:
        assert kwargs['in_channels'] == 3, 'in_channels must be 3 whem pretrained is True'
        print('load from imagenet')
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']), strict=False)
    return model


def resnet34(pretrained=True, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        assert kwargs['in_channels'] == 3, 'in_channels must be 3 whem pretrained is True'
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']), strict=False)
    return model


def resnet50(pretrained=True, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        assert kwargs['in_channels'] == 3, 'in_channels must be 3 whem pretrained is True'
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']), strict=False)
    return model


def deformable_resnet50(pretrained=True, **kwargs):
    """Constructs a ResNet-50 model with deformable conv.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], dcn=dict(deformable_groups=1), **kwargs)
    if pretrained:
        assert kwargs['in_channels'] == 3, 'in_channels must be 3 whem pretrained is True'
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']), strict=False)
    return model


def resnet101(pretrained=True, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        assert kwargs['in_channels'] == 3, 'in_channels must be 3 whem pretrained is True'
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']), strict=False)
    return model


def resnet152(pretrained=True, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        assert kwargs['in_channels'] == 3, 'in_channels must be 3 whem pretrained is True'
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']), strict=False)
    return model


if __name__ == '__main__':
    import torch

    x = torch.zeros(2, 3, 640, 640)
    net = deformable_resnet50(pretrained=False)
    y = net(x)
    for u in y:
        print(u.shape)

    print(net.out_channels)
