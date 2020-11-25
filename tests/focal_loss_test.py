import numpy as np
import pytest
import torch

from models.focal_loss import FocalLoss


def test_focal_loss():
    alpha, gamma = 0.25, 2
    fl = FocalLoss(alpha=alpha, gamma=gamma)
    logits = torch.from_numpy(np.asarray([[0.05, 0.3, -2], [0.5, -0.01, 0.03]], dtype=np.float32))
    target = torch.from_numpy(np.asarray([[0, 1, 0], [0, 0, 1]], dtype=np.float32))
    computed_loss = fl(pred_logits=logits, target=target)
    loss = torch.sum(computed_loss).item()

    target_loss = 0.
    for batch, batch_tgt in zip(logits, target):
        probs = torch.sigmoid(batch)
        for p, t in zip(probs, batch_tgt):
            pt = p if t else 1. - p
            alpha_t = alpha if t else 1. - alpha
            target_loss += -1. * alpha_t * ((1. - pt) ** gamma) * torch.log(pt)
    target_loss = target_loss.item()

    assert loss == pytest.approx(target_loss, 1e-3)
