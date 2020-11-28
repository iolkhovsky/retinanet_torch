import torch
import torch.nn as nn
import pytorch_lightning as pl

from models.mobilenet_v2 import SSDBackboneMobilenetv2
from models.ssd_predictor import SSDPredictor
from models.bbox_codec import FasterRCNNBoxCoder
from models.anchor_generator import AnchorGenerator
from models.ssd_loss import SSDLoss


class SSDMobilenet2(nn.Module):

    def __init__(self, anchors_cnt=6, classes_cnt=21):
        super(SSDMobilenet2, self).__init__()
        self.anchors, self.classes = anchors_cnt, classes_cnt
        self.backbone = SSDBackboneMobilenetv2(alpha=1., pretrained=True, requires_grad=True)
        self.predictor_heads = [
            SSDPredictor(in_channels=32, classes_cnt=classes_cnt, anchors_cnt=anchors_cnt),
            SSDPredictor(in_channels=96, classes_cnt=classes_cnt, anchors_cnt=anchors_cnt),
            SSDPredictor(in_channels=320, classes_cnt=classes_cnt, anchors_cnt=anchors_cnt),
            SSDPredictor(in_channels=480, classes_cnt=classes_cnt, anchors_cnt=anchors_cnt),
            SSDPredictor(in_channels=640, classes_cnt=classes_cnt, anchors_cnt=anchors_cnt),
            SSDPredictor(in_channels=640, classes_cnt=classes_cnt, anchors_cnt=anchors_cnt, kernel=1, pad=0)
        ]

    def forward(self, x):
        feature_maps = self.backbone(x)
        return [predictor(feature_map) for predictor, feature_map in zip(self.predictor_heads, feature_maps)]

    def __str__(self):
        feature_maps = len(self.classification_heads)
        return f"SSD_Mobilenetv2_{feature_maps}fm_{self.classes}c_{self.anchors}a"


class SSDLightning(pl.LightningDataModule):

    def __init__(self, anchors_cnt=6, classes_cnt=21):
        super().__init__()
        self.ssd = SSDMobilenet2(anchors_cnt=anchors_cnt, classes_cnt=classes_cnt)
        self.box_coder = FasterRCNNBoxCoder()
        self.anchor_gen = AnchorGenerator()
        self.criterion = SSDLoss(box_codec=self.box_coder, classes_cnt=classes_cnt, anchors_cnt=anchors_cnt)

    def forward(self, x):
        pass

    def training_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
