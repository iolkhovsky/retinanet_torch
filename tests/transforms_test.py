import numpy as np
import pytest

from utils.transforms import *


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


def test_normalization():
    img = np.random.randint(0, 255, size=(480, 640, 3))
    norm_img = normalize_image(img)
    denorm_img = denormalize_image(norm_img)
    assert np.sum(norm_img - denorm_img) == 0


if __name__ == "__main__":
    test_ndarray_cyx2yxc()
    test_ndarray_yxc2cyx()
    test_normalization()
