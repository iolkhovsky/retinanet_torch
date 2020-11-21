import torch.nn as nn


class SSDRegressionHead(nn.Module):

    def __init__(self, in_channels, anchors_cnt, kernel=3, pad=1):
        super(SSDRegressionHead, self).__init__()
        self.conv = nn.Conv2d(in_channels, 4 * anchors_cnt, kernel_size=(kernel, kernel), stride=(1, 1),
                              padding=(pad, pad), bias=False)

    def forward(self, x):
        return self.conv(x)


class SSDClassificationHead(nn.Module):

    def __init__(self, in_channels, classes_cnt, anchors_cnt, kernel=3, pad=1):
        super(SSDClassificationHead, self).__init__()
        self.conv = nn.Conv2d(in_channels, classes_cnt * anchors_cnt, kernel_size=(kernel, kernel), stride=(1, 1),
                              padding=(pad, pad), bias=False)

    def forward(self, x):
        return self.conv(x)
