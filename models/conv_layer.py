import torch
import torch.nn as nn


class ConvBnRelu(nn.Module):

    def __init__(self, in_chan, out_chan, kernel=3, stride=2, pad=1):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.Conv2d(in_chan, out_chan, kernel_size=(kernel, kernel), stride=(stride, stride),
                               padding=(pad, pad), bias=False)
        self.bn = nn.BatchNorm2d(out_chan, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.act = nn.ReLU6(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def load_weights(self, pars):
        assert len(pars) == 3
        for par in pars:
            assert type(par) == torch.nn.Parameter
        self.conv.weight, self.bn.weight, self.bn.bias = pars

    def enable_grad(self, enable):
        assert type(enable) == bool
        self.conv.weight.requires_grad = enable
        self.bn.weight.requires_grad = enable
        self.bn.bias.requires_grad = enable
