import numpy as np
import pytest
import torch

from utils.bounding_box import compute_iou, intersection_over_union
from utils.bounding_box import x1y1x2y2_to_xywh, xywh_to_x1y1x2y2
from utils.bounding_box import xcyc_to_xywh, xywh_to_xcyc


def test_intersection_over_union():
    box1, box2 = (1, 1, 2, 2), (2, 2, 2, 2)
    assert intersection_over_union(box1, box2) == pytest.approx(1. / 7., 1e-3)
    box2 = (3, 3, 1, 4)
    assert intersection_over_union(box1, box2) == pytest.approx(0., 1e-3)
    box2 = box1
    assert intersection_over_union(box1, box2) == pytest.approx(1., 1e-3)


def test_x1y1x2y2_to_xywh():
    box1 = torch.from_numpy(np.asarray([3, 4, 5, 6]))
    target = torch.from_numpy(np.asarray([3, 4, 2, 2]))
    res = x1y1x2y2_to_xywh(torch.unsqueeze(box1, 0))[0]
    assert torch.allclose(res, target)


def test_xywh_to_x1y1x2y2():
    box1 = torch.from_numpy(np.asarray([3, 4, 5, 6]))
    target = torch.from_numpy(np.asarray([3, 4, 8, 10]))
    res = xywh_to_x1y1x2y2(torch.unsqueeze(box1, 0))[0]
    assert torch.allclose(res, target)


def test_xcyc_to_xywh():
    box1 = torch.from_numpy(np.asarray([3, 4, 5, 6], dtype=np.float32))
    target = torch.from_numpy(np.asarray([0.5, 1., 5, 6], dtype=np.float32))
    res = xcyc_to_xywh(torch.unsqueeze(box1, 0))[0]
    assert torch.allclose(res, target)


def test_xywh_to_xcyc():
    box1 = torch.from_numpy(np.asarray([3, 4, 5, 6], dtype=np.float32))
    target = torch.from_numpy(np.asarray([5.5, 7., 5, 6], dtype=np.float32))
    res = xywh_to_xcyc(torch.unsqueeze(box1, 0))[0]
    assert torch.allclose(res, target)


def test_compute_iou():
    boxes1 = [(1, 1, 2, 2), (0, 0, 10, 10)]
    boxes2 = [(5, 6, 7, 8)]
    boxes1_tensors = torch.from_numpy(np.asarray(boxes1))
    boxes2_tensors = torch.from_numpy(np.asarray(boxes2))
    ious = compute_iou(boxes1_tensors, boxes2_tensors)

    for i, box1 in enumerate(boxes1):
        for j, box2 in enumerate(boxes2):
            target = intersection_over_union(box1, box2)
            assert ious[i, j] == pytest.approx(target, 0.1)
