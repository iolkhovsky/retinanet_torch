import pytest
import torch

from models.retinanet import RetinanetMobilenet2, RetinanetLightning


def test_retinanet():
    batch_size = 4
    aspect_ratios = [0.5, 1., 2.]
    scales = [2 ** x for x in [0, 1. / 3., 2. / 3.]]
    feature_map_sizes = [38, 19, 10, 5, 3, 1]
    anchors_cnt = len(aspect_ratios) * len(scales)
    classes_cnt = 21
    ssd = RetinanetMobilenet2(anchors_cnt, classes_cnt)
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


def test_retinanet_tl():
    classes_cnt = 21
    ssd = RetinanetLightning(classes_cnt=classes_cnt)
    batch_size = 4
    x = torch.rand(batch_size, 3, 300, 300)
    res = ssd.forward(x)
    assert len(res) == batch_size
    for img_idx, (clf_scores, bboxes) in enumerate(res):
        assert type(clf_scores) == list
        assert len(clf_scores)
        assert len(clf_scores[0]) == classes_cnt
        assert type(bboxes) == list
        assert len(bboxes)
        assert len(bboxes[0]) == 4


if __name__ == "__main__":
    test_retinanet()
    test_retinanet_tl()
