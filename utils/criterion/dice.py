import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_coeff(input: torch.Tensor, target: torch.Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    if input.dim() == 2 and reduce_batch_first:
        raise ValueError(f'Dice: asked to reduce batch but got tensor without batch dimension (shape {input.shape})')

    if input.dim() == 2 or reduce_batch_first:
        inter = torch.dot(input.reshape(-1), target.reshape(-1))
        sets_sum = torch.sum(input) + torch.sum(target)
        if sets_sum.item() == 0:
            sets_sum = 2 * inter

        return (2 * inter + epsilon) / (sets_sum + epsilon)
    else:
        # compute and average metric for each batch element
        dice = 0
        for i in range(input.shape[0]):
            dice += dice_coeff(input[i, ...], target[i, ...])
        return dice / input.shape[0]

def multiclass_dice_coeff(input: torch.Tensor, target: torch.Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all classes
    assert input.size() == target.size()
    dice = 0
    for channel in range(input.shape[1]):
        dice += dice_coeff(input[:, channel, ...], target[:, channel, ...], reduce_batch_first, epsilon)

    return dice / input.shape[1]

class DiceLoss(nn.Module):

    def __init__(self, multiclass: bool = True, num_classes: int = 2):
        super().__init__()
        self.multiclass = multiclass
        self.num_classes = num_classes
    
    def forward(self, input: torch.Tensor, target: torch.Tensor):
        
        input = F.softmax(input, dim=1).float()
        target = F.one_hot(target, self.num_classes).permute(0, 3, 1, 2).float()

        assert input.size() == target.size()
        fn = multiclass_dice_coeff if self.multiclass else dice_coeff
        return 1 - fn(input, target, reduce_batch_first=True)

