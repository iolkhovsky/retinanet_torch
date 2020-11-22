import numpy as np
import pytest
import torch

from models.ssd_loss import SSDRegressionLoss, FocalLoss


def test_ssd_regression_loss():
    preds = np.asarray([(0, 0, 2, 2), (1, 1, 4, 5), (2, 3, 4, 5), (3, 4, 5, 6)], dtype=np.float32)
    target = np.asarray([(0.5, 0.7, 2.3, 2.4), (3, 3, 10, 7), (3, 3, 5, 8), (3, 4, 5, 6)], dtype=np.float32)
    target_loss = []
    for p, t in zip(preds.flatten(), target.flatten()):
        if abs(p - t) < 1.:
            target_loss.append(0.5 * abs(p-t) ** 2)
        else:
            target_loss.append(abs(p - t) - 0.5)
    target_loss = np.asarray(target_loss).mean()
    pred_tensor = torch.from_numpy(preds).view(2, 2, 4)
    target_tensor = torch.from_numpy(target).view(2, 2, 4)
    criterion = SSDRegressionLoss()
    loss = criterion.forward(predicted_boxes=pred_tensor, target_boxes=target_tensor)
    assert loss.item() == pytest.approx(target_loss, 1e-3)


def test_focal_loss():
    criterion = FocalLoss(logits=True)
    pred = torch.from_numpy(np.asarray([[0.2, 9., 3.], [-4, 2, 10]], dtype=np.float32))
    target = torch.from_numpy(np.asarray([[0, 1, 0], [0, 1, 0]], dtype=np.float32))
    loss = criterion(inputs=pred, targets=target)
    print(loss)


if __name__ == "__main__":
    test_focal_loss()
