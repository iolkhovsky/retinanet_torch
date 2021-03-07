import torch


def intersect_bboxes(box1, box2):
    empty_box = 0, 0, 0, 0
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    intersection_x0 = max(x1, x2)
    intersection_x1 = min(x1 + w1, x2 + w2)
    if intersection_x1 <= intersection_x0:
        return empty_box
    intersection_y0 = max(y1, y2)
    intersection_y1 = min(y1 + h1, y2 + h2)
    if intersection_y1 <= intersection_y0:
        return empty_box
    return intersection_x0, intersection_y0, (intersection_x1 - intersection_x0), (intersection_y1 - intersection_y0)


def intersection_over_union(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    inter_x, inter_y, inter_w, inter_h = intersect_bboxes(box1, box2)
    intersection_area = inter_w * inter_h
    if intersection_area == 0:
        return 0.

    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - intersection_area
    return intersection_area / union_area if union_area else 0.


def x1y1x2y2_to_xywh(boxes):
    # (xmin, ymin, xmax, ymax) -> (x, y, w, h)
    return torch.cat([boxes[..., :2], boxes[..., 2:] - boxes[..., :2]], dim=-1)


def xywh_to_x1y1x2y2(boxes):
    # (x, y, w, h) -> (xmin, ymin, xmax, ymax)
    return torch.cat([boxes[..., :2], boxes[..., :2] + boxes[..., 2:]], dim=-1)


def xcyc_to_xywh(boxes):
    # (xc, yc, w, h) -> (x, y, w, h)
    return torch.cat([boxes[..., :2] - boxes[..., 2:] / 2.0, boxes[..., 2:]], dim=-1)


def xywh_to_xcyc(boxes):
    # (x, y, w, h) -> (xc, yc, w, h)
    return torch.cat([boxes[..., :2] + boxes[..., 2:] / 2.0, boxes[..., 2:]], dim=-1)


def compute_iou(boxes1, boxes2):
    boxes1_corners = xywh_to_x1y1x2y2(boxes1)
    boxes2_corners = xywh_to_x1y1x2y2(boxes2)
    lu = torch.maximum(boxes1_corners[:, None, :2], boxes2_corners[:, :2])
    rd = torch.minimum(boxes1_corners[:, None, 2:], boxes2_corners[:, 2:])
    intersection = torch.clamp(rd - lu, min=0.)
    intersection_area = intersection[:, :, 0] * intersection[:, :, 1]
    boxes1_area = boxes1[:, 2] * boxes1[:, 3]
    boxes2_area = boxes2[:, 2] * boxes2[:, 3]
    union_area = torch.clamp(boxes1_area[:, None] + boxes2_area - intersection_area, min=1e-8)
    return torch.clamp(intersection_area / union_area, 0.0, 1.0)
