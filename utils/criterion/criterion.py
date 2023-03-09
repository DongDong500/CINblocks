from .crossentropy import CrossEntropyLoss
from .entropydice   import EntropyDiceLoss
from .regentropydice import RegEntropyDiceLoss
from .dice import DiceLoss


def entropydice(args):

    return EntropyDiceLoss(
        update_weight=False
    )

def red(args):
    """Regression + CrossEntropy + Dice loss function for the proposed ESnet.
    """
    return RegEntropyDiceLoss(
        update_weight=False
    )
