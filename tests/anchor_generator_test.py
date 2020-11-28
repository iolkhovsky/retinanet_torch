import cv2
import numpy as np
import pytest

from models.anchor_generator import AnchorGenerator


def test_anchor_generator():
    aspect_ratios = [0.5, 1., 2.]
    scales = [2 ** x for x in [0, 1./3., 2./3.]]
    gen = AnchorGenerator(aspect_ratios=aspect_ratios, scales=scales)
    img_size = 300
    for map_size in [38, 19, 10, 5, 3, 1]:
        draw_scale = 5
        image = np.zeros(shape=(img_size * draw_scale, img_size * draw_scale, 3), dtype=np.uint8)
        anchors = gen.generate(img_size, map_size)
        assert anchors.shape == (map_size * map_size * len(scales) * len(aspect_ratios), 4)
        for x, y, w, h in anchors:
            x, y, w, h = int(x), int(y), int(w), int(h)
            cv2.rectangle(image, (x * draw_scale, y * draw_scale), ((x + w) * draw_scale, (y + h) * draw_scale),
                          (0, 255, 0), 1)
        cv2.imwrite(f"tests/output/map{map_size}.jpg", image)
