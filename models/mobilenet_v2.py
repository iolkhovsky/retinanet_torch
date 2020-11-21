import torch.nn as nn
from torchvision import models

from models.conv_layer import SimpleConvolution, BottleneckConvolution, DepthwiseSeparableConvolution


class MobilenetV2FeatureExtractor(nn.Module):

    def __init__(self, alpha=1., pretrained=True, requires_grad=True):
        super(MobilenetV2FeatureExtractor, self).__init__()

        self.conv0 = SimpleConvolution(in_chan=3, out_chan=int(32 * alpha), kernel=3, stride=2, pad=1)
        self.bottleneck1 = BottleneckConvolution(in_chan=int(32 * alpha), t_factor=1, out_chan=int(16 * alpha),
                                                 stride=1)
        self.bottleneck2 = BottleneckConvolution(in_chan=int(16 * alpha), t_factor=6, out_chan=int(24 * alpha),
                                                 stride=2)
        self.bottleneck3 = BottleneckConvolution(in_chan=int(24 * alpha), t_factor=6, out_chan=int(24 * alpha),
                                                 stride=1)
        self.bottleneck4 = BottleneckConvolution(in_chan=int(24 * alpha), t_factor=6, out_chan=int(32 * alpha),
                                                 stride=2)
        self.bottleneck5 = BottleneckConvolution(in_chan=int(32 * alpha), t_factor=6, out_chan=int(32 * alpha),
                                                 stride=1)
        self.bottleneck6 = BottleneckConvolution(in_chan=int(32 * alpha), t_factor=6, out_chan=int(32 * alpha),
                                                 stride=1)
        self.bottleneck7 = BottleneckConvolution(in_chan=int(32 * alpha), t_factor=6, out_chan=int(64 * alpha),
                                                 stride=2)
        self.bottleneck8 = BottleneckConvolution(in_chan=int(64 * alpha), t_factor=6, out_chan=int(64 * alpha),
                                                 stride=1)
        self.bottleneck9 = BottleneckConvolution(in_chan=int(64 * alpha), t_factor=6, out_chan=int(64 * alpha),
                                                 stride=1)
        self.bottleneck10 = BottleneckConvolution(in_chan=int(64 * alpha), t_factor=6, out_chan=int(64 * alpha),
                                                  stride=1)
        self.bottleneck11 = BottleneckConvolution(in_chan=int(64 * alpha), t_factor=6, out_chan=int(96 * alpha),
                                                  stride=1)
        self.bottleneck12 = BottleneckConvolution(in_chan=int(96 * alpha), t_factor=6, out_chan=int(96 * alpha),
                                                  stride=1)
        self.bottleneck13 = BottleneckConvolution(in_chan=int(96 * alpha), t_factor=6, out_chan=int(96 * alpha),
                                                  stride=1)
        self.bottleneck14 = BottleneckConvolution(in_chan=int(96 * alpha), t_factor=6, out_chan=int(160 * alpha),
                                                  stride=2)
        self.bottleneck15 = BottleneckConvolution(in_chan=int(160 * alpha), t_factor=6, out_chan=int(160 * alpha),
                                                  stride=1)
        self.bottleneck16 = BottleneckConvolution(in_chan=int(160 * alpha), t_factor=6, out_chan=int(160 * alpha),
                                                  stride=1)
        self.bottleneck17 = BottleneckConvolution(in_chan=int(160 * alpha), t_factor=6, out_chan=int(320 * alpha),
                                                  stride=1)
        self.conv18 = SimpleConvolution(in_chan=int(320 * alpha), out_chan=int(1280 * alpha), kernel=1, stride=1, pad=0)

        if pretrained:
            self.load_pretrained_weights()
        self.enable_grad(enable=requires_grad)

    def forward(self, x):
        x = self.conv0(x)
        x = self.bottleneck1(x)
        x = self.bottleneck2(x)
        x = self.bottleneck3(x)
        x = self.bottleneck4(x)
        x = self.bottleneck5(x)
        x = self.bottleneck6(x)
        x = self.bottleneck7(x)
        x = self.bottleneck8(x)
        x = self.bottleneck9(x)
        x = self.bottleneck10(x)
        x = self.bottleneck11(x)
        x = self.bottleneck12(x)
        x = self.bottleneck13(x)
        x = self.bottleneck14(x)
        x = self.bottleneck15(x)
        x = self.bottleneck16(x)
        x = self.bottleneck17(x)
        x = self.conv18(x)
        return x

    def load_pretrained_weights(self):
        pretrained_params = list(models.mobilenet_v2(pretrained=True).parameters())
        self.conv0.load_weights(pretrained_params[:3])
        self.bottleneck1.load_weights(pretrained_params[3:9])
        self.bottleneck2.load_weights(pretrained_params[9:18])
        self.bottleneck3.load_weights(pretrained_params[18:27])
        self.bottleneck4.load_weights(pretrained_params[27:36])
        self.bottleneck5.load_weights(pretrained_params[36:45])
        self.bottleneck6.load_weights(pretrained_params[45:54])
        self.bottleneck7.load_weights(pretrained_params[54:63])
        self.bottleneck8.load_weights(pretrained_params[63:72])
        self.bottleneck9.load_weights(pretrained_params[72:81])
        self.bottleneck10.load_weights(pretrained_params[81:90])
        self.bottleneck11.load_weights(pretrained_params[90:99])
        self.bottleneck12.load_weights(pretrained_params[99:108])
        self.bottleneck13.load_weights(pretrained_params[108:117])
        self.bottleneck14.load_weights(pretrained_params[117:126])
        self.bottleneck15.load_weights(pretrained_params[126:135])
        self.bottleneck16.load_weights(pretrained_params[135:144])
        self.bottleneck17.load_weights(pretrained_params[144:153])
        self.conv18.load_weights(pretrained_params[153:156])

    def enable_grad(self, enable):
        assert type(enable) == bool
        for par in self.parameters():
            par.requires_grad = enable


class SSDBackboneMobilenetv2(nn.Module):

    def __init__(self, alpha=1., pretrained=True, requires_grad=True):
        super(SSDBackboneMobilenetv2, self).__init__()

        self.conv0 = SimpleConvolution(in_chan=3, out_chan=int(32 * alpha), kernel=3, stride=2, pad=1)
        self.bottleneck1 = BottleneckConvolution(in_chan=int(32 * alpha), t_factor=1, out_chan=int(16 * alpha),
                                                 stride=1)
        self.bottleneck2 = BottleneckConvolution(in_chan=int(16 * alpha), t_factor=6, out_chan=int(24 * alpha),
                                                 stride=2)
        self.bottleneck3 = BottleneckConvolution(in_chan=int(24 * alpha), t_factor=6, out_chan=int(24 * alpha),
                                                 stride=1)
        self.bottleneck4 = BottleneckConvolution(in_chan=int(24 * alpha), t_factor=6, out_chan=int(32 * alpha),
                                                 stride=2)
        self.bottleneck5 = BottleneckConvolution(in_chan=int(32 * alpha), t_factor=6, out_chan=int(32 * alpha),
                                                 stride=1)
        self.bottleneck6 = BottleneckConvolution(in_chan=int(32 * alpha), t_factor=6, out_chan=int(32 * alpha),
                                                 stride=1)
        self.bottleneck7 = BottleneckConvolution(in_chan=int(32 * alpha), t_factor=6, out_chan=int(64 * alpha),
                                                 stride=2)
        self.bottleneck8 = BottleneckConvolution(in_chan=int(64 * alpha), t_factor=6, out_chan=int(64 * alpha),
                                                 stride=1)
        self.bottleneck9 = BottleneckConvolution(in_chan=int(64 * alpha), t_factor=6, out_chan=int(64 * alpha),
                                                 stride=1)
        self.bottleneck10 = BottleneckConvolution(in_chan=int(64 * alpha), t_factor=6, out_chan=int(64 * alpha),
                                                  stride=1)
        self.bottleneck11 = BottleneckConvolution(in_chan=int(64 * alpha), t_factor=6, out_chan=int(96 * alpha),
                                                  stride=1)
        self.bottleneck12 = BottleneckConvolution(in_chan=int(96 * alpha), t_factor=6, out_chan=int(96 * alpha),
                                                  stride=1)
        self.bottleneck13 = BottleneckConvolution(in_chan=int(96 * alpha), t_factor=6, out_chan=int(96 * alpha),
                                                  stride=1)
        self.bottleneck14 = BottleneckConvolution(in_chan=int(96 * alpha), t_factor=6, out_chan=int(160 * alpha),
                                                  stride=2)
        self.bottleneck15 = BottleneckConvolution(in_chan=int(160 * alpha), t_factor=6, out_chan=int(160 * alpha),
                                                  stride=1)
        self.bottleneck16 = BottleneckConvolution(in_chan=int(160 * alpha), t_factor=6, out_chan=int(160 * alpha),
                                                  stride=1)
        self.bottleneck17 = BottleneckConvolution(in_chan=int(160 * alpha), t_factor=6, out_chan=int(320 * alpha),
                                                  stride=1)

        self.dwconv18 = DepthwiseSeparableConvolution(in_chan=int(320 * alpha), out_chan=int(480 * alpha), kernel=3,
                                                      stride=2, pad=1)
        self.dwconv19 = DepthwiseSeparableConvolution(in_chan=int(480 * alpha), out_chan=int(640 * alpha), kernel=3,
                                                      stride=1, pad=0)
        self.dwconv20 = DepthwiseSeparableConvolution(in_chan=int(640 * alpha), out_chan=int(640 * alpha), kernel=3,
                                                      stride=1, pad=0)

        if pretrained:
            self.load_pretrained_weights()
        self.enable_grad(requires_grad)

    def load_pretrained_weights(self):
        pretrained_params = list(models.mobilenet_v2(pretrained=True).parameters())
        self.conv0.load_weights(pretrained_params[:3])
        self.bottleneck1.load_weights(pretrained_params[3:9])
        self.bottleneck2.load_weights(pretrained_params[9:18])
        self.bottleneck3.load_weights(pretrained_params[18:27])
        self.bottleneck4.load_weights(pretrained_params[27:36])
        self.bottleneck5.load_weights(pretrained_params[36:45])
        self.bottleneck6.load_weights(pretrained_params[45:54])
        self.bottleneck7.load_weights(pretrained_params[54:63])
        self.bottleneck8.load_weights(pretrained_params[63:72])
        self.bottleneck9.load_weights(pretrained_params[72:81])
        self.bottleneck10.load_weights(pretrained_params[81:90])
        self.bottleneck11.load_weights(pretrained_params[90:99])
        self.bottleneck12.load_weights(pretrained_params[99:108])
        self.bottleneck13.load_weights(pretrained_params[108:117])
        self.bottleneck14.load_weights(pretrained_params[117:126])
        self.bottleneck15.load_weights(pretrained_params[126:135])
        self.bottleneck16.load_weights(pretrained_params[135:144])
        self.bottleneck17.load_weights(pretrained_params[144:153])

    def forward(self, x):
        x = self.conv0(x)
        x = self.bottleneck1(x)
        x = self.bottleneck2(x)
        x = self.bottleneck3(x)
        x = self.bottleneck4(x)
        x = self.bottleneck5(x)
        out_0 = self.bottleneck6(x)
        x = self.bottleneck7(out_0)
        x = self.bottleneck8(x)
        x = self.bottleneck9(x)
        x = self.bottleneck10(x)
        x = self.bottleneck11(x)
        x = self.bottleneck12(x)
        out_1 = self.bottleneck13(x)
        x = self.bottleneck14(out_1)
        x = self.bottleneck15(x)
        x = self.bottleneck16(x)
        out_2 = self.bottleneck17(x)
        out_3 = self.dwconv18(out_2)
        out_4 = self.dwconv19(out_3)
        out_5 = self.dwconv20(out_4)
        return out_0, out_1, out_2, out_3, out_4, out_5

    def enable_grad(self, enable):
        assert type(enable) == bool
        for par in self.parameters():
            par.requires_grad = enable
