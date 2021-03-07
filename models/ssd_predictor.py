import numpy as np
import torch.nn as nn


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


class SSDPredictor(nn.Module):

    def __init__(self, in_channels, classes_cnt, anchors_cnt, kernel=3, pad=1):
        super(SSDPredictor, self).__init__()
        self.clf = SSDClassificationHead(in_channels=in_channels, anchors_cnt=anchors_cnt, classes_cnt=classes_cnt,
                                         kernel=kernel, pad=pad)
        self.regr = SSDRegressionHead(in_channels=in_channels, anchors_cnt=anchors_cnt, kernel=kernel, pad=pad)

    def forward(self, x):
        return self.clf(x), self.regr(x)
