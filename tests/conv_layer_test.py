import pytest
import torch

from models.conv_layer import SimpleConvolution, DepthwiseSeparableConvolution


def test_simple_conv_layer():
    in_channels, out_channels = 3, 9
    kernel_sz, stride, pad = 3, 1, 1
    batch_sz, ysz, xsz = 4, 9, 9
    layer = SimpleConvolution(in_chan=in_channels, out_chan=out_channels, kernel=kernel_sz, stride=stride, pad=pad)
    x = torch.rand(batch_sz, in_channels, ysz, xsz)
    out = layer(x)
    assert len(out.shape) == len(x.shape)
    assert out.shape[0] == x.shape[0] == batch_sz
    assert out.shape[1] == out_channels
    assert out.shape[2] == (ysz + 2 * pad - 2 * (kernel_sz // 2)) // stride
    assert out.shape[3] == (xsz + 2 * pad - 2 * (kernel_sz // 2)) // stride


def test_grad_simple_conv_layer():
    layer = SimpleConvolution(in_chan=1, out_chan=3)
    layer.enable_grad(True)
    assert layer.conv.weight.requires_grad
    assert layer.bn.weight.requires_grad
    assert layer.bn.bias.requires_grad
    layer.enable_grad(False)
    assert not layer.conv.weight.requires_grad
    assert not layer.bn.weight.requires_grad
    assert not layer.bn.bias.requires_grad


def test_load_simple_conv_layer():
    in_channels, out_channels = 4, 9
    kernel_sz, stride, pad = 3, 1, 1
    layer = SimpleConvolution(in_chan=in_channels, out_chan=out_channels, kernel=kernel_sz, stride=stride, pad=pad)
    conv_weights = torch.nn.Parameter(torch.rand(out_channels, in_channels, kernel_sz, kernel_sz))
    bn_weights = torch.nn.Parameter(torch.rand(out_channels))
    bn_bias = torch.nn.Parameter(torch.rand(out_channels))
    assert not torch.equal(layer.conv.weight, conv_weights)
    assert not torch.equal(layer.bn.weight, bn_weights)
    assert not torch.equal(layer.bn.bias, bn_bias)
    layer.load_weights([conv_weights, bn_weights, bn_bias])
    assert torch.equal(layer.conv.weight, conv_weights)
    assert torch.equal(layer.bn.weight, bn_weights)
    assert torch.equal(layer.bn.bias, bn_bias)


def test_depthwise_sep_convolution():
    in_channels, out_channels = 3, 9
    kernel_sz, stride, pad = 3, 1, 1
    batch_sz, ysz, xsz = 4, 9, 9
    layer = DepthwiseSeparableConvolution(in_chan=in_channels, out_chan=out_channels, kernel=kernel_sz, stride=stride, pad=pad)
    x = torch.rand(batch_sz, in_channels, ysz, xsz)
    out = layer(x)
    assert len(out.shape) == len(x.shape)
    assert out.shape[0] == x.shape[0] == batch_sz
    assert out.shape[1] == out_channels
    assert out.shape[2] == (ysz + 2 * pad - 2 * (kernel_sz // 2)) // stride
    assert out.shape[3] == (xsz + 2 * pad - 2 * (kernel_sz // 2)) // stride


if __name__ == "__main__":
    test_simple_conv_layer()
    test_grad_simple_conv_layer()
    test_load_simple_conv_layer()
    test_depthwise_sep_convolution()
