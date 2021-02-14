import torch
import torch.nn as nn


class SimpleConvolution(nn.Module):

    def __init__(self, in_chan, out_chan, kernel=3, stride=2, pad=1, requires_grad=True):
        super(SimpleConvolution, self).__init__()
        self.conv = nn.Conv2d(in_chan, out_chan, kernel_size=(kernel, kernel), stride=(stride, stride),
                               padding=(pad, pad), bias=False)
        self.bn = nn.BatchNorm2d(out_chan, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.act = nn.ReLU6(inplace=True)
        self.enable_grad(requires_grad)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def load_weights(self, pars):
        assert len(pars) == 3
        for par in pars:
            assert type(par) == torch.nn.Parameter
        self.conv.weight, self.bn.weight, self.bn.bias = pars

    def enable_grad(self, enable):
        assert type(enable) == bool
        for par in self.parameters():
            par.requires_grad = enable


class DepthwiseSeparableConvolution(nn.Module):

    def __init__(self, in_chan, out_chan, kernel, stride, pad):
        super(DepthwiseSeparableConvolution, self).__init__()
        self.conv_dw = nn.Conv2d(in_chan, in_chan, kernel_size=(kernel, kernel), stride=(stride, stride),
                                 groups=in_chan, bias=False, padding=(pad, pad))
        self.bn_dw = nn.BatchNorm2d(in_chan, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.act_dw = nn.ReLU6(inplace=True)
        self.conv_pw_out = nn.Conv2d(in_chan, out_chan, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bn_pw_out = nn.BatchNorm2d(out_chan, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.act_pw = nn.ReLU6(inplace=True)

    def forward(self, x):
        x = self.act_dw(self.bn_dw(self.conv_dw(x)))
        x = self.act_pw(self.bn_pw_out(self.conv_pw_out(x)))
        return x


class BottleneckConvolution(nn.Module):

    def __init__(self, in_chan, out_chan, t_factor=6., stride=1, requires_grad=True):
        super(BottleneckConvolution, self).__init__()
        self.in_channels, self.out_channels = in_chan, out_chan
        self.expansion_factor = t_factor
        self.stride = stride
        inner_size = int(self.in_channels * t_factor)

        self.conv_pw = nn.Conv2d(in_chan, inner_size, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bn_pw = nn.BatchNorm2d(inner_size, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.act_pw = nn.ReLU6(inplace=True)

        self.conv_dw = nn.Conv2d(inner_size, inner_size, kernel_size=(3, 3), stride=(self.stride, self.stride),
                                 groups=inner_size, bias=False, padding=(1, 1))
        self.bn_dw = nn.BatchNorm2d(inner_size, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.act_dw = nn.ReLU6(inplace=True)

        self.conv_pw_out = nn.Conv2d(inner_size, out_chan, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bn_pw_out = nn.BatchNorm2d(out_chan, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.enable_grad(requires_grad)

    def forward(self, x):
        in_act = x
        x = self.act_pw(self.bn_pw(self.conv_pw(x))) if self.expansion_factor != 1 else x
        x = self.act_dw(self.bn_dw(self.conv_dw(x)))
        x = self.bn_pw_out(self.conv_pw_out(x))
        out_act = torch.add(x, in_act) if (self.in_channels == self.out_channels) and (self.stride == 1) else x
        return out_act

    def load_weights(self, pars):
        assert len(pars) == (9 if self.expansion_factor != 1. else 6)
        for par in pars:
            assert type(par) == torch.nn.Parameter
        if self.expansion_factor != 1.:
            self.conv_pw.weight, self.bn_pw.weight, self.bn_pw.bias = pars[:3]
            self.conv_dw.weight, self.bn_dw.weight, self.bn_dw.bias = pars[3:6]
            self.conv_pw_out.weight, self.bn_pw_out.weight, self.bn_pw_out.bias = pars[6:9]
        else:
            self.conv_dw.weight, self.bn_dw.weight, self.bn_dw.bias = pars[:3]
            self.conv_pw_out.weight, self.bn_pw_out.weight, self.bn_pw_out.bias = pars[3:6]

    def enable_grad(self, enable):
        assert type(enable) == bool
        for par in self.parameters():
            par.requires_grad = enable
