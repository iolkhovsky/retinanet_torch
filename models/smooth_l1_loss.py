import torch.nn as nn


class SmoothL1Loss(nn.Module):

    def __init__(self):
        super(SmoothL1Loss, self).__init__()
        self.criterion = nn.SmoothL1Loss(beta=1.)

    def forward(self, predicted_boxes, target_boxes):
        """
        :param predicted_boxes: [N, 4]
        :param target_boxes: [N, 4]
        :return: Smooth L1 loss, averaged per batch
        """
        return self.criterion(input=predicted_boxes.flatten(), target=target_boxes.flatten())
