import numpy as np
import torch


def ndarray_yxc2cyx(arr):
    assert type(arr) == np.ndarray
    return np.swapaxes(np.swapaxes(arr, 1, 2), 0, 1)


def ndarray_cyx2yxc(arr):
    assert type(arr) == np.ndarray
    return np.swapaxes(np.swapaxes(arr, 0, 1), 1, 2)


def one_hot_encoding(labels, num_classes):
    """
    :param labels: [N,]
    :param num_classes: M
    :return: one-hot encded
    """
    y = torch.eye(num_classes)
    return y[labels]
