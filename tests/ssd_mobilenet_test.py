import pytest
import torch

from models.ssd_mobilenet_v2 import SSDMobilenet2


def test_ssd_mobilenet():
    batch_size = 4
    aspect_ratios = [0.5, 1., 2.]
    scales = [2 ** x for x in [0, 1. / 3., 2. / 3.]]
    feature_map_sizes = [38, 19, 10, 5, 3, 1]
    anchors_cnt = len(aspect_ratios) * len(scales)
    classes_cnt = 21
    ssd = SSDMobilenet2(anchors_cnt, classes_cnt)
    x = torch.rand(batch_size, 3, 300, 300)
    predictions = ssd(x)
    assert len(predictions) == len(feature_map_sizes)
    for head_prediction, map_size in zip(predictions, feature_map_sizes):
        assert len(head_prediction) == 2
        classification, regression = head_prediction
        clf_output_size = list(classification.size())
        rgr_output_size = list(regression.size())
        target_clf_size = [batch_size, anchors_cnt * classes_cnt, map_size, map_size]
        target_rgr_size = [batch_size, anchors_cnt * 4, map_size, map_size]
        assert clf_output_size == target_clf_size
        assert target_rgr_size == rgr_output_size


if __name__ == "__main__":
    test_ssd_mobilenet()