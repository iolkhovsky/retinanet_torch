import numpy as np
import pytest

from utils.transforms import ndarray_cyx2yxc, ndarray_yxc2cyx


def test_ndarray_cyx2yxc():
    csz, ysz, xsz = 3, 4, 5
    arr = np.zeros(shape=(csz, ysz, xsz), dtype=np.uint8)
    c, y, x, probe = 2, 1, 4, 77
    arr[c, y, x] = probe
    transformed_array = ndarray_cyx2yxc(arr)
    assert transformed_array.shape == (ysz, xsz, csz)
    assert transformed_array[y, x, c] == probe


def test_ndarray_yxc2cyx():
    ysz, xsz, csz = 5, 7, 3
    arr = np.zeros(shape=(ysz, xsz, csz), dtype=np.uint8)
    c, y, x, probe = 2, 1, 4, 70
    arr[y, x, c] = probe
    transformed_array = ndarray_yxc2cyx(arr)
    assert transformed_array.shape == (csz, ysz, xsz)
    assert transformed_array[c, y, x] == probe


if __name__ == "__main__":
    test_ndarray_cyx2yxc()
    test_ndarray_yxc2cyx()
