import numpy as np
import pytest
import torch

from models.bbox_codec import FasterRCNNBoxCoder


def test_fasterrcnn_boxcoder():
    coder = FasterRCNNBoxCoder()
    boxes = torch.from_numpy(np.asarray([(1, 2, 3, 4), (2, 2, 4, 4)], dtype=np.float32))
    anchors = torch.from_numpy(np.asarray([(1, 2, 3, 4), (1, 3, 5, 7)], dtype=np.float32))
    encoded_boxes = coder.encode(boxes, anchors)
    decoded_boxes = coder.decode(encoded_boxes, anchors)
    for i in range(len(decoded_boxes)):
        assert torch.allclose(boxes[i], decoded_boxes[i])
