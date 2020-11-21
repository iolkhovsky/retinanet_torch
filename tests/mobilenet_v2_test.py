import numpy as np
import pytest
import torch
import torchvision.models as torch_models

from models.mobilenet_v2 import MobilenetV2FeatureExtractor, SSDBackboneMobilenetv2


def test_ssd_mobilenet_backbone():
    fext = SSDBackboneMobilenetv2(alpha=1., pretrained=False, requires_grad=False)
    batch_cnt = 3
    x = torch.randn(batch_cnt, 3, 300, 300)
    out = fext(x)
    assert type(out) == tuple
    assert len(out) == 6
    assert out[0].shape == (batch_cnt, 32, 38, 38)
    assert out[1].shape == (batch_cnt, 96, 19, 19)
    assert out[2].shape == (batch_cnt, 320, 10, 10)
    assert out[3].shape == (batch_cnt, 480, 5, 5)
    assert out[4].shape == (batch_cnt, 640, 3, 3)
    assert out[5].shape == (batch_cnt, 640, 1, 1)


def test_mobilenet_fext():
    fext = MobilenetV2FeatureExtractor(alpha=1., pretrained=False, requires_grad=False)
    batch_cnt = 3
    x = torch.randn(batch_cnt, 3, 224, 224)
    out = fext(x)
    assert type(out) == torch.Tensor
    assert out.shape == (batch_cnt, 1280, 7, 7)


def test_pretrained_mobilenet():
    fext = MobilenetV2FeatureExtractor(alpha=1., pretrained=True, requires_grad=False)
    reference_model = torch_models.mobilenet_v2(pretrained=True)
    batch_cnt = 3
    x = torch.randn(batch_cnt, 3, 224, 224)
    ref_out = reference_model.features(x)
    out = fext(x)
    assert np.all(torch.eq(out, ref_out).numpy())
