import torch
import torch.nn as nn

from utils.bounding_box import xywh_to_xcyc, xcyc_to_xywh


EPSILON = 1e-6


class FasterRCNNBoxCoder(nn.Module):

    def __init__(self, scale_factors=[10., 10., 5., 5.]):
        super(FasterRCNNBoxCoder, self).__init__()
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

        anchors_height = anchors_h + EPSILON
        anchors_width = anchors_w + EPSILON
        boxes_height = boxes_h + EPSILON
        boxes_width = boxes_w + EPSILON

        tx = (boxes_x - anchors_x) / anchors_width
        ty = (boxes_y - anchors_y) / anchors_height
        tw = torch.log(boxes_width / anchors_width)
        th = torch.log(boxes_height / anchors_height)

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
        tx_scaled = tx.clone()
        ty_scaled = ty.clone()
        tw_scaled = tw.clone()
        th_scaled = th.clone()
        if self._scale_factors:
            tx_scaled /= self._scale_factors[0]
            ty_scaled /= self._scale_factors[1]
            tw_scaled /= self._scale_factors[2]
            th_scaled /= self._scale_factors[3]
        w = torch.exp(tw_scaled) * anchors_w
        h = torch.exp(th_scaled) * anchors_h
        ycenter = ty_scaled * anchors_h + anchors_y
        xcenter = tx_scaled * anchors_w + anchors_x
        decoded = torch.cat([torch.unsqueeze(xcenter, 0), torch.unsqueeze(ycenter, 0), torch.unsqueeze(w, 0),
                             torch.unsqueeze(h, 0)], dim=0)
        decoded = torch.transpose(decoded, 0, 1)
        return xcyc_to_xywh(decoded)
