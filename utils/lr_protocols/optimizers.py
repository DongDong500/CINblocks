import torch


def sgd(param, args):
    
    return torch.optim.SGD(
        params=param,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
