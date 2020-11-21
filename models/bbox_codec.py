import torch
import torch.nn as nn

from utils.bounding_box import xywh_to_xcyc, xcyc_to_xywh


EPSILON = 1e-6


class FasterRCNNBoxCoder(nn.Module):

    def __init__(self, scale_factors=None):
        if scale_factors:
            assert len(scale_factors) == 4
            for scalar in scale_factors:
                assert scalar > 0
        self._scale_factors = scale_factors

    def encode(self, boxes, anchors):
        centered_boxes = xywh_to_xcyc(boxes)
        centered_anchors = xywh_to_xcyc(anchors)

        boxes_x, boxes_y, boxes_w, boxes_h = torch.unbind(torch.transpose(centered_boxes, 0, 1))
        anchors_x, anchors_y, anchors_w, anchors_h = torch.unbind(torch.transpose(centered_anchors, 0, 1))

        anchors_h += EPSILON
        anchors_w += EPSILON
        boxes_h += EPSILON
        boxes_w += EPSILON

        tx = (boxes_x - anchors_x) / anchors_w
        ty = (boxes_y - anchors_y) / anchors_h
        tw = torch.log(boxes_w / anchors_w)
        th = torch.log(boxes_h / anchors_h)

        if self._scale_factors:
            tx *= self._scale_factors[0]
            ty *= self._scale_factors[1]
            tw *= self._scale_factors[2]
            th *= self._scale_factors[3]
        encoded = torch.cat([torch.unsqueeze(tx, 0), torch.unsqueeze(ty, 0), torch.unsqueeze(tw, 0),
                             torch.unsqueeze(th, 0)], dim=0)
        return torch.transpose(encoded, 0, 1)

    def decode(self, rel_codes, anchors):
        centered_anchors = xywh_to_xcyc(anchors)
        anchors_x, anchors_y, anchors_w, anchors_h = torch.unbind(torch.transpose(centered_anchors, 0, 1))
        tx, ty, tw, th = torch.unbind(torch.transpose(rel_codes, 0, 1))

        if self._scale_factors:
            tx /= self._scale_factors[0]
            ty /= self._scale_factors[1]
            tw /= self._scale_factors[2]
            th /= self._scale_factors[3]
        w = torch.exp(tw) * anchors_w
        h = torch.exp(th) * anchors_h
        ycenter = ty * anchors_h + anchors_y
        xcenter = tx * anchors_w + anchors_x
        decoded = torch.cat([torch.unsqueeze(xcenter, 0), torch.unsqueeze(ycenter, 0), torch.unsqueeze(w, 0),
                             torch.unsqueeze(h, 0)], dim=0)
        decoded = torch.transpose(decoded, 0, 1)
        return xcyc_to_xywh(decoded)
