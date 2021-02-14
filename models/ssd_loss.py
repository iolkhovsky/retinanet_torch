import torch
import torch.nn as nn

from utils.bounding_box import compute_iou
from models.focal_loss import FocalLoss
from models.smooth_l1_loss import SmoothL1Loss


class SSDLoss(nn.Module):

    def __init__(self, box_codec, anchors_cnt=6, classes_cnt=21, classification_weight=1., regression_weight=1.):
        super(SSDLoss, self).__init__()
        self.anchors_cnt = anchors_cnt
        self.classes_cnt = classes_cnt
        self.box_codec = box_codec
        self.classifcation_w = classification_weight
        self.regression_w = regression_weight
        self.regression_criterion = SmoothL1Loss()
        self.classification_criterion = FocalLoss()

    def forward(self, classification_preds, boxes_preds, anchors, target_boxes, target_labels, iou_thresh=0.5):
        classification_preds = classification_preds.view(-1, self.classes_cnt)
        boxes_preds = boxes_preds.view(-1, 4)
        anchors = anchors.view(-1, 4)
        target_boxes = target_boxes.view(-1, 4)
        target_labels = torch.flatten(target_labels)

        iou_matrix = compute_iou(target_boxes, anchors)
        max_iou_for_anchors, target_ids_for_anchors = torch.max(iou_matrix, dim=0)
        positive_anchors_mask = max_iou_for_anchors >= iou_thresh
        positive_anchors_cnt = torch.sum(torch.where(positive_anchors_mask, 1, 0))

        classification_targets = torch.zeros(size=classification_preds.size())
        classification_targets[positive_anchors_mask] = \
            nn.functional.one_hot(target_labels[target_ids_for_anchors][positive_anchors_mask]).float()
        classification_loss = self.classification_criterion(classification_preds, classification_targets)
        classification_loss = torch.sum(classification_loss) / positive_anchors_cnt

        predicted_positive_boxes = boxes_preds[positive_anchors_mask]
        max_iou_for_targets, anchor_ids_for_targets = torch.max(iou_matrix, dim=1)
        target_match_mask = max_iou_for_targets >= iou_thresh
        matched_target_boxes = target_boxes[target_match_mask]
        anchors_for_matched_targets = anchors[anchor_ids_for_targets][target_match_mask]
        encoded_target_boxes = self.box_codec.encode(matched_target_boxes, anchors_for_matched_targets)
        regression_loss = self.regression_criterion(predicted_positive_boxes, encoded_target_boxes)

        weighed_class_loss = classification_loss * self.classifcation_w
        weighed_regr_loss = regression_loss * self.regression_w

        return weighed_class_loss + weighed_regr_loss, weighed_class_loss, weighed_regr_loss
