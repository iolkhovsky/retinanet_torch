import numpy as np
import pytest
import torch

from models.bbox_codec import FasterRCNNBoxCoder
from models.retinanet_loss import RetinaNetLoss


def test_ssd_loss():
    coder = FasterRCNNBoxCoder()
    criterion = RetinaNetLoss(box_codec=coder, anchors_cnt=2, classes_cnt=3)

    anchors = torch.from_numpy(
        np.asarray([(10, 10, 100, 20), (10, 50, 20, 100), (100, 100, 100, 20), (100, 10, 30, 100)]))
    ground_truth_boxes = torch.from_numpy(
        np.asarray([(8, 8, 90, 20), (95, 10, 25, 95)]))
    ground_truth_labels = torch.from_numpy(np.asarray([1, 2]))

    pred_boxes = torch.from_numpy(
        np.asarray([(10, 10, 100, 20), (5, 55, 25, 80), (90, 105, 115, 15), (97, 9, 30, 110)]))
    pred_boxes_encoded = coder.encode(pred_boxes, anchors)
    pred_labels = torch.from_numpy(
        np.asarray([(0.2, 0.55, 0.01), (0.01, 0.2, 0.15), (0.015, 0.4, 0.2), (0.1, 0.2, 0.6)]))

    total, classification, regression = criterion(classification_preds=pred_labels,
                                                  boxes_preds=pred_boxes_encoded,
                                                  anchors=anchors,
                                                  target_boxes=ground_truth_boxes,
                                                  target_labels=ground_truth_labels)
