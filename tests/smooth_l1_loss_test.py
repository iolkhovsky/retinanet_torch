import numpy as np
import pytest
import torch

from models.smooth_l1_loss import SmoothL1Loss


def test_smooth_l1_loss():
    preds = np.asarray([(0, 0, 2, 2), (1, 1, 4, 5), (2, 3, 4, 5), (3, 4, 5, 6)], dtype=np.float32)
    target = np.asarray([(0.5, 0.7, 2.3, 2.4), (3, 3, 10, 7), (3, 3, 5, 8), (3, 4, 5, 6)], dtype=np.float32)
    target_loss = []
    for p, t in zip(preds.flatten(), target.flatten()):
        if abs(p - t) < 1.:
            target_loss.append(0.5 * abs(p-t) ** 2)
        else:
            target_loss.append(abs(p - t) - 0.5)
    target_loss = np.asarray(target_loss).sum()
    pred_tensor = torch.from_numpy(preds).view(2, 2, 4)
    target_tensor = torch.from_numpy(target).view(2, 2, 4)
    criterion = SmoothL1Loss()
    loss = criterion.forward(predicted_boxes=pred_tensor, target_boxes=target_tensor)
    assert loss.item() == pytest.approx(target_loss, 1e-3)
