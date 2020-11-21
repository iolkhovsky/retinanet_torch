import pytest
import torch

from models.ssd_mobilenet_v2 import SSDMobilenet2


def test_ssd_mobilenet():
    anchors_cnt, classes_cnt = 3, 21
    ssd = SSDMobilenet2(anchors_cnt, classes_cnt)
    x = torch.rand(3, 300, 300)
    out = ssd(x)
