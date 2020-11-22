import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.transforms import one_hot_encoding


class SSDRegressionLoss(nn.Module):

    def __init__(self):
        super(SSDRegressionLoss, self).__init__()
        self.criterion = nn.SmoothL1Loss(beta=1.)

    def forward(self, predicted_boxes, target_boxes):
        """
        :param predicted_boxes: [N, 4]
        :param target_boxes: [N, 4]
        :return: Smooth L1 loss, averaged per batch
        """
        return self.criterion(input=predicted_boxes.flatten(), target=target_boxes.flatten())


class SSDClassificationLoss(nn.Module):

    def __init__(self, total_classes, alpha=0.25, gamma=2.0):
        super(SSDClassificationLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.total_classes = total_classes

    def forward(self, predicted_confidences, traget_ids):
        """
        :param predicted_confidences: [N, classes_cnt]
        :param traget_ids: [N]
        :return:
        """
        target_confidences = one_hot_encoding(traget_ids, self.total_classes)



class SSDLoss(nn.Module):

    def __init__(self):
        pass

    def forward(self, x):
        pass