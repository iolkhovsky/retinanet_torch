import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class FocalLoss(nn.Module):

    def __init__(self, alpha=0.25, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        assert isinstance(alpha, (float, int))
        self.alpha = torch.Tensor([alpha, 1-alpha])
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, pred, target):
        target = target.view(-1, 1)

        logpt = F.log_softmax(pred)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=pred.data.type():
                self.alpha = self.alpha.type_as(pred.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()
