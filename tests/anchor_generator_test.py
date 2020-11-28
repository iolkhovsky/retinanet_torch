import pytest

from models.anchor_generator import AnchorGenerator


def test_anchor_generator():
    aspect_ratios = [0.5, 1., 2.]
    scales = [2 ** x for x in [0, 1./3., 2./3.]]
    gen = AnchorGenerator(aspect_ratios=aspect_ratios, scales=scales)
    img_size, map_sz = 300, 38
    anchors = gen.generate(img_size, map_sz)
    assert anchors.shape == (map_sz * map_sz * len(scales) * len(aspect_ratios), 4)
