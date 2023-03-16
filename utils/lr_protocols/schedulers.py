import torch


def steplr(optimizer, args):

    return torch.optim.lr_scheduler.StepLR(
        optimizer=optimizer,
        step_size=args.step_size,
        gamma=args.gamma,
        verbose=args.verbose
    )
