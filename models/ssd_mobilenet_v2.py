import torch.nn as nn

from models.ssd_head import SSDClassificationHead, SSDRegressionHead
from models.mobilenet_v2 import SSDBackboneMobilenetv2


class SSDMobilenet2(nn.Module):

    def __init__(self, anchors_cnt=6, classes_cnt=21):
        self.anchors, self.classes = anchors_cnt, classes_cnt
        self.backbone = SSDBackboneMobilenetv2(alpha=1., pretrained=True, requires_grad=True)
        self.classification_heads = [
            SSDClassificationHead(in_channels=32, classes_cnt=classes_cnt, anchors_cnt=anchors_cnt),
            SSDClassificationHead(in_channels=96, classes_cnt=classes_cnt, anchors_cnt=anchors_cnt),
            SSDClassificationHead(in_channels=320, classes_cnt=classes_cnt, anchors_cnt=anchors_cnt),
            SSDClassificationHead(in_channels=480, classes_cnt=classes_cnt, anchors_cnt=anchors_cnt),
            SSDClassificationHead(in_channels=640, classes_cnt=classes_cnt, anchors_cnt=anchors_cnt),
            SSDClassificationHead(in_channels=640, classes_cnt=classes_cnt, anchors_cnt=anchors_cnt, kernel=1, pad=0)
        ]
        self.regression_heads = [
            SSDRegressionHead(in_channels=32, anchors_cnt=anchors_cnt),
            SSDRegressionHead(in_channels=96, anchors_cnt=anchors_cnt),
            SSDRegressionHead(in_channels=320, anchors_cnt=anchors_cnt),
            SSDRegressionHead(in_channels=480, anchors_cnt=anchors_cnt),
            SSDRegressionHead(in_channels=640, anchors_cnt=anchors_cnt),
            SSDRegressionHead(in_channels=640, anchors_cnt=anchors_cnt, kernel=1, pad=0)
        ]

    def forward(self):
        pass

    def __str__(self):
        feature_maps = len(self.classification_heads)
        return f"SSD_Mobilenetv2_{feature_maps}fm_{self.classes}c_{self.anchors}a"
