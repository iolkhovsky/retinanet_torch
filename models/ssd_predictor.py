import numpy as np
import torch.nn as nn

from models.conv_layer import SimpleConvolution


class SSDRegressionHead(nn.Module):

    def __init__(self, in_channels, anchors_cnt, kernel=3, pad=1):
        super(SSDRegressionHead, self).__init__()
        regression_channels = 4 * anchors_cnt
        self.conv = nn.Conv2d(in_channels, regression_channels, kernel_size=(kernel, kernel), stride=(1, 1),
                              padding=(pad, pad))

    def forward(self, x):
        return self.conv(x)


class SSDClassificationHead(nn.Module):

    def __init__(self, in_channels, anchors_cnt, classes_cnt, kernel=3, pad=1, pi=0.01):
        super(SSDClassificationHead, self).__init__()
        classification_channels = classes_cnt * anchors_cnt
        self.conv = nn.Conv2d(in_channels, classification_channels, kernel_size=(kernel, kernel), stride=(1, 1),
                              padding=(pad, pad), bias=True)
        initial_bias = -1. * np.log((1. - pi) / pi)
        self.conv.bias.data.fill_(initial_bias)

    def forward(self, x):
        return self.conv(x)


class RetinaNetPredictorSubnet(nn.Module):

    def __init__(self, in_channels, depth=256, layers=4, kernel=3, pad=1):
        super(RetinaNetPredictorSubnet, self).__init__()
        self.layers = [SimpleConvolution(in_chan=in_channels, out_chan=depth, kernel=kernel, stride=1, pad=pad)]
        for i in range(1, layers):
            self.layers.append(SimpleConvolution(in_chan=depth, out_chan=depth, kernel=kernel, stride=1, pad=pad))
        self.layers = nn.ModuleList(self.layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class RetinaNetPredictor(nn.Module):

    def __init__(self, in_channels, classes_cnt, anchors_cnt, kernel=3, pad=1, subnet_depth=256, subnet_layers=4):
        super(RetinaNetPredictor, self).__init__()
        self.clf_subnet = RetinaNetPredictorSubnet(in_channels=in_channels, depth=subnet_depth, layers=subnet_layers, kernel=kernel, pad=pad)
        self.clf_head = SSDClassificationHead(in_channels=subnet_depth, anchors_cnt=anchors_cnt, classes_cnt=classes_cnt,
                                              kernel=kernel, pad=pad)
        self.regr_subnet = RetinaNetPredictorSubnet(in_channels=in_channels, depth=subnet_depth, layers=subnet_layers, kernel=kernel, pad=pad)
        self.regr_head = SSDRegressionHead(in_channels=subnet_depth, anchors_cnt=anchors_cnt, kernel=kernel, pad=pad)

    def forward(self, x):
        clf_subnet_out = self.clf_subnet(x)
        clf_head_output = self.clf_head(clf_subnet_out)
        regr_subnet_out = self.regr_subnet(x)
        regr_head_output = self.regr_head(regr_subnet_out)
        return clf_head_output, regr_head_output
