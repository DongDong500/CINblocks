import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Optional
from torch.nn.modules.loss import _WeightedLoss


def dice_coeff(
    input: torch.Tensor, 
    target: torch.Tensor, 
    reduce_batch_first: bool = False, 
    epsilon=1e-6
):
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

def multiclass_dice_coeff(
    input: torch.Tensor, 
    target: torch.Tensor, 
    reduce_batch_first: bool = False, 
    epsilon=1e-6
):
    # Average of Dice coefficient for all classes
    assert input.size() == target.size()
    dice = 0
    for channel in range(input.shape[1]):
        dice += dice_coeff(input[:, channel, ...], target[:, channel, ...], reduce_batch_first, epsilon)

    return dice / input.shape[1]

class RegEntropyDiceLoss(_WeightedLoss):

    __constants__ = ['ignore_index', 'reduction', 'label_smoothing']
    ignore_index: int
    label_smoothing: float

    def __init__(
        self, weight: Optional[torch.Tensor] = None, 
        size_average=None, 
        ignore_index: int = -100,
        reduce=None, 
        reduction: str = 'mean', 
        label_smoothing: float = 0.0 , 
        multiclass: bool = True, 
        update_weight: bool = True, 
        num_classes: int = 2
    ) -> None:
        super(RegEntropyDiceLoss, self).__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        self.multiclass = multiclass
        self.update_weight = update_weight
        self.num_classes = num_classes
    
    def _update_weight(self, target, weight: Optional[Tensor] = None):
        weights = target.sum() / (target.shape[0] * target.shape[1] * target.shape[2])
        if weights < 0 or weights > 1:
            raise Exception (f'weights: {weights} for cross entropy is wrong')
        self.weight = torch.tensor([weights, 1 - weights], dtype=torch.float32)

    def forward(self, input: dict, target: dict) -> torch.Tensor:
        
        # outputs from model
        logits = input['image']
        center = input['center']

        # ground truth value
        mask = target['mask']
        anchor = target['center']

        if self.update_weight and self.num_classes == 2:
            self._update_weight(mask, )

        # CrossEntropy loss
        ce = F.cross_entropy(
            logits,
            mask, 
            weight=self.weight,
            ignore_index=self.ignore_index, 
            reduction=self.reduction,
            label_smoothing=self.label_smoothing
        )

        # Dice loss
        logits = F.softmax(logits, dim=1).float()
        mask = F.one_hot(mask, self.num_classes).permute(0, 3, 1, 2).float()

        assert logits.size() == mask.size()

        fn = multiclass_dice_coeff if self.multiclass else dice_coeff
        dl = 1 - fn(logits, mask, reduce_batch_first=True)

        # MSE loss
        mse = F.mse_loss(center, anchor, reduction=self.reduction)

        return ce + dl + mse
