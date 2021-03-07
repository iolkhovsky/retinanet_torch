import torch
import torch.nn as nn

from utils.bounding_box import compute_iou
from models.smooth_l1_loss import SmoothL1Loss


class SSDLoss(nn.Module):

    def __init__(self, box_codec, anchors_cnt=6, classes_cnt=21, classification_weight=1., regression_weight=1.,
                 hard_negatives_ratio=3.):
        super(SSDLoss, self).__init__()
        self.anchors_cnt = anchors_cnt
        self.classes_cnt = classes_cnt
        self.box_codec = box_codec
        self.classifcation_w = classification_weight
        self.regression_w = regression_weight
        self.regression_criterion = SmoothL1Loss()
        self.classification_criterion = nn.CrossEntropyLoss(reduction="sum")
        assert hard_negatives_ratio > 0
        self.hard_negatives_ratio = hard_negatives_ratio

    def forward(self, classification_preds, boxes_preds, anchors, target_boxes, target_labels, iou_thresh=0.5):
        assert isinstance(classification_preds, (list, torch.Tensor))
        assert isinstance(boxes_preds, (list, torch.Tensor))
        assert isinstance(anchors, (list, torch.Tensor))
        if type(classification_preds) == list:
            classification_preds = torch.stack(classification_preds)
        if type(boxes_preds) == list:
            boxes_preds = torch.stack(boxes_preds)
        assert len(classification_preds) == len(boxes_preds) == len(anchors)
        assert len(target_boxes) == len(target_labels)
        assert iou_thresh > 0.1
        classification_preds = classification_preds.view(-1, self.classes_cnt)
        boxes_preds = boxes_preds.view(-1, 4)
        anchors = anchors.view(-1, 4)
        target_boxes = target_boxes.view(-1, 4)
        target_labels = torch.flatten(target_labels)

        iou_matrix = compute_iou(target_boxes, anchors)
        max_iou_for_anchors, target_ids_for_anchors = torch.max(iou_matrix, dim=0)
        positive_anchors_mask = max_iou_for_anchors >= iou_thresh
        positive_anchors_cnt = torch.sum(positive_anchors_mask.int())
        normalization_cnt = torch.clamp(positive_anchors_cnt, min=1)

        positive_anchors_targets = target_labels[target_ids_for_anchors[positive_anchors_mask]]
        positive_classification_loss = self.classification_criterion(
            input=classification_preds[positive_anchors_mask],
            target=positive_anchors_targets)
        negative_anchors_mask = torch.logical_not(positive_anchors_mask)
        background_preds = classification_preds[negative_anchors_mask]
        background_max_confs, _ = torch.max(background_preds[:, 1:], dim=1)
        _, hard_negatives = zip(*sorted(zip(background_max_confs, background_preds), reverse=True, key=lambda x: x[0]))
        hard_negatives = torch.stack(hard_negatives)
        hard_negatives_cnt = int(self.hard_negatives_ratio * normalization_cnt)
        negative_classification_loss = self.classification_criterion(
            input=hard_negatives[:hard_negatives_cnt],
            target=torch.as_tensor([0] * hard_negatives_cnt))

        classification_loss = (positive_classification_loss + negative_classification_loss) / normalization_cnt

        regression_loss = 0.
        if positive_anchors_cnt:
            encoded_boxes_preds = self.box_codec.encode(boxes_preds, anchors)
            predicted_positive_boxes = encoded_boxes_preds[positive_anchors_mask]
            target_boxes_ids = target_ids_for_anchors[positive_anchors_mask]
            target_boxes_for_positives = target_boxes[target_boxes_ids]
            anchors_for_positives = anchors[positive_anchors_mask]
            encoded_target_boxes = self.box_codec.encode(target_boxes_for_positives, anchors_for_positives)
            regression_loss = self.regression_criterion(predicted_positive_boxes, encoded_target_boxes)

        weighed_class_loss = classification_loss * self.classifcation_w
        weighed_regr_loss = regression_loss * self.regression_w

        return weighed_class_loss + weighed_regr_loss, weighed_class_loss, weighed_regr_loss
