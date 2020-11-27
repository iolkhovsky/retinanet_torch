import torch
import torch.nn as nn

from utils.bounding_box import compute_iou
from models.focal_loss import FocalLoss
from models.smooth_l1_loss import SmoothL1Loss


class SSDLoss(nn.Module):

    def __init__(self, box_codec, anchors_cnt=6, classes_cnt=21):
        super(SSDLoss).__init__()
        self.anchors_cnt = anchors_cnt
        self.classes_cnt = classes_cnt
        self.box_codec = box_codec
        self.regression_criterion = SmoothL1Loss()
        self.classification_criterion = FocalLoss()

    def forward(self, classification_preds, boxes_preds, anchors, target_boxes, target_labels):
        iou_matrix = compute_iou(target_boxes, anchors)
        max_iou = torch.max(iou_matrix, dim=0)

        positive_anchors_indices = max_iou >= 0.5
        positive_anchors_cnt = torch.sum(torch.where(positive_anchors_indices, 1, 0))

        classification_targets = nn.functional.one_hot(target_labels)

        classification_loss = self.classification_criterion(classification_preds, )

        encoded_targets = self.box_codec.encode(target_labels, anchors)
        pass