'''
Hourglass network inserted in the pre-activated Resnet
Use lr=0.01 for current version
(c) YANG, Wei
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url


__all__ = ['HourglassNet', 'hg']


model_urls = {
    'hg1': 'https://github.com/anibali/pytorch-stacked-hourglass/releases/download/v0.0.0/bearpaw_hg1-ce125879.pth',
    'hg2': 'https://github.com/anibali/pytorch-stacked-hourglass/releases/download/v0.0.0/bearpaw_hg2-15e342d9.pth',
    'hg8': 'https://github.com/anibali/pytorch-stacked-hourglass/releases/download/v0.0.0/bearpaw_hg8-90e5d470.pth',
}


class Conv2d_WS(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias)

    def forward(self, x):
        # return super(Conv2d, self).forward(x)
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,
                                  keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return F.conv2d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, activation, norm, conv, stride=1, downsample=None):
        super(Bottleneck, self).__init__()

        self.bn1 = norm(inplanes)
        self.conv1 = conv(inplanes, planes, kernel_size=1, bias=True)
        self.bn2 = norm(planes)
        self.conv2 = conv(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=True)
        self.bn3 = norm(planes)
        self.conv3 = conv(planes, planes * 2, kernel_size=1, bias=True)
        self.relu = activation
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out


class Hourglass(nn.Module):
    def __init__(self, block, num_blocks, planes, depth, activation, norm, conv, compress_ratio):
        super(Hourglass, self).__init__()
        depth = depth // compress_ratio

        self.depth = depth
        self.block = block
        self.activation = activation
        self.norm = norm
        self.conv = conv
        self.hg = self._make_hour_glass(block, num_blocks, planes, depth)

    def _make_residual(self, block, num_blocks, planes):
        layers = []
        for i in range(0, num_blocks):
            layers.append(block(planes*block.expansion, planes, self.activation, self.norm, self.conv))
        return nn.Sequential(*layers)

    def _make_hour_glass(self, block, num_blocks, planes, depth):
        hg = []
        for i in range(depth):
            res = []
            for j in range(3):
                res.append(self._make_residual(block, num_blocks, planes))
            if i == 0:
                res.append(self._make_residual(block, num_blocks, planes))
            hg.append(nn.ModuleList(res))
        return nn.ModuleList(hg)

    def _hour_glass_forward(self, n, x):
        up1 = self.hg[n-1][0](x)
        low1 = F.max_pool2d(x, 2, stride=2)
        low1 = self.hg[n-1][1](low1)

        if n > 1:
            low2 = self._hour_glass_forward(n-1, low1)
        else:
            low2 = self.hg[n-1][3](low1)
        low3 = self.hg[n-1][2](low2)
        up2 = F.interpolate(low3, scale_factor=2)
        out = up1 + up2
        return out

    def forward(self, x):
        return self._hour_glass_forward(self.depth, x)


class HourglassNet(nn.Module):
    '''Hourglass model from Newell et al ECCV 2016'''
    def __init__(self, block, num_stacks=2, num_blocks=4, num_classes=16, activation=None, norm_type=None, use_weight_std=False, compress_ratio=1):
        super(HourglassNet, self).__init__()
        
        print(activation)
        if activation == 'relu':
            self.relu = nn.ReLU(inplace=True)
        elif activation == 'gelu':
            self.relu = nn.GELU()
        elif activation == 'leakyrelu':
            self.relu = nn.LeakyReLU()
            
        print(norm_type)
        if norm_type == 'batch':
            self.norm = nn.BatchNorm2d
        elif norm_type == 'instance':
            self.norm = nn.InstanceNorm2d
        elif norm_type == 'group':
            self.norm = lambda num_channels: nn.GroupNorm(32, num_channels)
            
        print(f'Use weight standardization: {use_weight_std}')
        if use_weight_std:
            self.conv = Conv2d_WS
        else:
            self.conv = nn.Conv2d

        self.inplanes = 64
        self.num_feats = 128
        self.num_stacks = num_stacks
        self.conv1 = self.conv(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=True)
        self.bn1 = self.norm(self.inplanes)
            
        self.layer1 = self._make_residual(block, self.inplanes, 1)
        self.layer2 = self._make_residual(block, self.inplanes, 1)
        self.layer3 = self._make_residual(block, self.num_feats, 1)
        self.maxpool = nn.MaxPool2d(2, stride=2)

        # build hourglass modules
        ch = self.num_feats*block.expansion
        hg, res, fc, score, fc_, score_ = [], [], [], [], [], []
        for i in range(num_stacks):
            if i == num_stacks - 1:
                hg.append(Hourglass(block, num_blocks, self.num_feats, 4, activation=self.relu, norm=self.norm, conv=self.conv, compress_ratio=compress_ratio))
            else:
                hg.append(Hourglass(block, num_blocks, self.num_feats, 4, activation=self.relu, norm=self.norm, conv=self.conv, compress_ratio=1))
            res.append(self._make_residual(block, self.num_feats, num_blocks))
            fc.append(self._make_fc(ch, ch))
            score.append(nn.Conv2d(ch, num_classes, kernel_size=1, bias=True))
            if i < num_stacks-1:
                fc_.append(nn.Conv2d(ch, ch, kernel_size=1, bias=True))
                score_.append(nn.Conv2d(num_classes, ch, kernel_size=1, bias=True))
        self.hg = nn.ModuleList(hg)
        self.res = nn.ModuleList(res)
        self.fc = nn.ModuleList(fc)
        self.score = nn.ModuleList(score)
        self.fc_ = nn.ModuleList(fc_)
        self.score_ = nn.ModuleList(score_)

    def _make_residual(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=True),
            )

        layers = []
        layers.append(block(self.inplanes, planes, self.relu, self.norm, self.conv, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, self.relu, self.norm))

        return nn.Sequential(*layers)

    def _make_fc(self, inplanes, outplanes):
        bn = self.norm(inplanes)
        conv = self.conv(inplanes, outplanes, kernel_size=1, bias=True)
        return nn.Sequential(
                conv,
                bn,
                self.relu,
            )

    def forward(self, x):
        out = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.maxpool(x)
        x = self.layer2(x)
        x = self.layer3(x)

        for i in range(self.num_stacks):
            y = self.hg[i](x)
            y = self.res[i](y)
            y = self.fc[i](y)
            score = self.score[i](y)
            out.append(score)
            if i < self.num_stacks-1:
                fc_ = self.fc_[i](y)
                score_ = self.score_[i](score)
                x = x + fc_ + score_

        return out


def hg(**kwargs):
    model = HourglassNet(Bottleneck, num_stacks=kwargs['num_stacks'], num_blocks=kwargs['num_blocks'],
                         num_classes=kwargs['num_classes'],
                         activation=kwargs['activation'],
                         norm_type=kwargs['norm'],
                         use_weight_std=kwargs['weight_std'],
                         compress_ratio=kwargs['compress_ratio'])
    return model


def _hg(**kwargs):
    model = hg(**kwargs)
    if kwargs['pretrained']:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=True,
                                              map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
    return model


def hg1(pretrained=False, progress=True, num_blocks=1, num_classes=16):
    return _hg('hg1', pretrained, progress, num_stacks=1, num_blocks=num_blocks,
               num_classes=num_classes)


def hg2(pretrained=False, progress=True, num_blocks=1, num_classes=16):
    return _hg('hg2', pretrained, progress, num_stacks=2, num_blocks=num_blocks,
               num_classes=num_classes)


def hg8(pretrained=False, progress=True, num_blocks=1, num_classes=16):
    return _hg('hg8', pretrained, progress, num_stacks=8, num_blocks=num_blocks,
               num_classes=num_classes)
