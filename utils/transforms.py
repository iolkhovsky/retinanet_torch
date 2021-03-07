import cv2
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


def pil_to_array(pil_img):
    return np.array(pil_img)


def pil_to_cv_img(pil_img):
    return cv2.cvtColor(pil_to_array(pil_img), cv2.COLOR_RGB2BGR)


def normalize_image(rgb_img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    rgb_img[:, :, 0] = (rgb_img[:, :, 0] - mean[0]) / std[0]
    rgb_img[:, :, 1] = (rgb_img[:, :, 1] - mean[1]) / std[1]
    rgb_img[:, :, 2] = (rgb_img[:, :, 2] - mean[2]) / std[2]
    return rgb_img


def denormalize_image(rgb_img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    rgb_img[:, :, 0] = rgb_img[:, :, 0] * std[0] + mean[0]
    rgb_img[:, :, 1] = rgb_img[:, :, 1] * std[1] + mean[1]
    rgb_img[:, :, 2] = rgb_img[:, :, 2] * std[2] + mean[2]
    return rgb_img
