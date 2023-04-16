"""
Creates a MobileNetV3 Model as defined in:
Andrew Howard, Mark Sandler, Grace Chu, Liang-Chieh Chen, Bo Chen, Mingxing Tan, Weijun Wang, Yukun Zhu, Ruoming Pang, Vijay Vasudevan, Quoc V. Le, Hartwig Adam. (2019).
Searching for MobileNetV3
arXiv preprint arXiv:1905.02244.
"""

import torch.nn as nn
import math
from utils.learning import freeze_params


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, _make_divisible(channel // reduction, 8)),
            nn.ReLU(inplace=True),
            nn.Linear(_make_divisible(channel // reduction, 8), channel),
            h_sigmoid())

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


def conv_3x3_bn(inp, oup, stride, norm_layer=nn.BatchNorm2d):
    return nn.Sequential(nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                         norm_layer(oup), h_swish())


def conv_1x1_bn(inp, oup, norm_layer=nn.BatchNorm2d):
    return nn.Sequential(nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                         norm_layer(oup), h_swish())


class InvertedResidual(nn.Module):
    def __init__(self,
                 inp,
                 hidden_dim,
                 oup,
                 kernel_size,
                 stride,
                 use_se,
                 use_hs,
                 dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        self.identity = stride == 1 and inp == oup

        if inp == hidden_dim:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim,
                          hidden_dim,
                          kernel_size,
                          stride, (kernel_size - 1) // 2 * dilation,
                          dilation=dilation,
                          groups=hidden_dim,
                          bias=False),
                norm_layer(hidden_dim),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # Squeeze-and-Excite
                SELayer(hidden_dim) if use_se else nn.Identity(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                norm_layer(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                norm_layer(hidden_dim),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(hidden_dim,
                          hidden_dim,
                          kernel_size,
                          stride, (kernel_size - 1) // 2 * dilation,
                          dilation=dilation,
                          groups=hidden_dim,
                          bias=False),
                norm_layer(hidden_dim),
                # Squeeze-and-Excite
                SELayer(hidden_dim) if use_se else nn.Identity(),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                norm_layer(oup),
            )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV3Large(nn.Module):
    def __init__(self,
                 output_stride=16,
                 norm_layer=nn.BatchNorm2d,
                 width_mult=1.,
                 freeze_at=0):
        super(MobileNetV3Large, self).__init__()
        """
        Constructs a MobileNetV3-Large model
        """
        cfgs = [
            # k, t, c, SE, HS, s
            [3, 1, 16, 0, 0, 1],
            [3, 4, 24, 0, 0, 2],
            [3, 3, 24, 0, 0, 1],
            [5, 3, 40, 1, 0, 2],
            [5, 3, 40, 1, 0, 1],
            [5, 3, 40, 1, 0, 1],
            [3, 6, 80, 0, 1, 2],
            [3, 2.5, 80, 0, 1, 1],
            [3, 2.3, 80, 0, 1, 1],
            [3, 2.3, 80, 0, 1, 1],
            [3, 6, 112, 1, 1, 1],
            [3, 6, 112, 1, 1, 1],
            [5, 6, 160, 1, 1, 2],
            [5, 6, 160, 1, 1, 1],
            [5, 6, 160, 1, 1, 1]
        ]
        self.cfgs = cfgs

        # building first layer
        input_channel = _make_divisible(16 * width_mult, 8)
        layers = [conv_3x3_bn(3, input_channel, 2, norm_layer)]
        # building inverted residual blocks
        block = InvertedResidual
        now_stride = 2
        rate = 1
        for k, t, c, use_se, use_hs, s in self.cfgs:
            if now_stride == output_stride:
                dilation = rate
                rate *= s
                s = 1
            else:
                dilation = 1
                now_stride *= s
            output_channel = _make_divisible(c * width_mult, 8)
            exp_size = _make_divisible(input_channel * t, 8)
            layers.append(
                block(input_channel, exp_size, output_channel, k, s, use_se,
                      use_hs, dilation, norm_layer))
            input_channel = output_channel

        self.features = nn.Sequential(*layers)
        self.conv = conv_1x1_bn(input_channel, exp_size, norm_layer)
        # building last several layers

        self._initialize_weights()

        feature_4x = self.features[0:4]
        feautre_8x = self.features[4:7]
        feature_16x = self.features[7:13]
        feature_32x = self.features[13:]

        self.stages = [feature_4x, feautre_8x, feature_16x, feature_32x]

        self.freeze(freeze_at)

    def forward(self, x):
        xs = []
        for stage in self.stages:
            x = stage(x)
            xs.append(x)
        xs[-1] = self.conv(xs[-1])
        return xs

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def freeze(self, freeze_at):
        if freeze_at >= 1:
            for m in self.stages[0][0]:
                freeze_params(m)

        for idx, stage in enumerate(self.stages, start=2):
            if freeze_at >= idx:
                freeze_params(stage)
