import torch.nn as nn


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False), nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)  # sq
        y = self.fc(y).view(b, c, 1, 1)  # ex

        return x * y.expand_as(x)  # scale


def norm(dim):
    return nn.GroupNorm(min(32, dim), dim)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False)


class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, reduction=16):
        super(SEBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.conv2 = conv3x3(planes, planes)
        self.norm1 = norm(planes)
        self.norm2 = norm(planes)
        self.norm3 = norm(planes)
        self.relu = nn.ReLU(inplace=True)
        self.se = SELayer(planes, reduction)
        if inplanes != planes:
            self.downsample = nn.Sequential(
                conv1x1(inplanes, planes, stride), norm(planes))
        else:
            self.downsample = lambda x: x

    def forward(self, x):
        residual = self.downsample(x)

        out = self.norm1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.norm3(out)
        out = self.se(out)

        out += residual

        return out


class SEBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, reduction=16):
        super(SEBottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.conv3 = conv1x1(planes, inplanes)
        self.norm1 = norm(inplanes)
        self.norm2 = norm(planes)
        self.norm3 = norm(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.se = SELayer(inplanes, reduction)
        if inplanes != planes:
            self.downsample = nn.Sequential(
                conv1x1(inplanes, planes, stride), norm(planes))
        else:
            self.downsample = lambda x: x

    def forward(self, x):
        residual = self.downsample(x)

        out = self.norm1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.se(out)

        out += residual

        return out
