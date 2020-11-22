import torch.nn as nn


class SSDPredictor(nn.Module):

    def __init__(self, in_channels, classes_cnt, anchors_cnt, kernel=3, pad=1):
        super(SSDPredictor, self).__init__()
        regression_channels = 4 * anchors_cnt
        classification_channels = classes_cnt * anchors_cnt
        self.conv = nn.Conv2d(in_channels, classification_channels + regression_channels, kernel_size=(kernel, kernel),
                              stride=(1, 1), padding=(pad, pad))

    def forward(self, x):
        return self.conv(x)
