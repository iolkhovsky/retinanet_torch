import torch.nn as nn


class SSDLoss(nn.Module):

    def __init__(self, anchors_cnt=6, classes_cnt=21):
        super(SSDLoss).__init__()
        self.anchors_cnt = anchors_cnt
        self.classes_cnt = classes_cnt

    def forward(self, classification_preds, boxes_preds, anchors, target_boxes, target_labels):
        pass