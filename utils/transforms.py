import numpy as np


def ndarray_yxc2cyx(arr):
    assert type(arr) == np.ndarray
    return np.swapaxes(np.swapaxes(arr, 1, 2), 0, 1)


def ndarray_cyx2yxc(arr):
    assert type(arr) == np.ndarray
    return np.swapaxes(np.swapaxes(arr, 0, 1), 1, 2)
