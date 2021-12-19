from torch import nn
from torch.nn import functional as F

class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=0.0000001):
        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()

        dice_loss = 1 - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')

        Dice_BCE = BCE + dice_loss

        return Dice_BCE

        # return Dice_BCE, BCE, dice_loss