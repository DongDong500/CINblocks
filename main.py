import os
import argparse
import datetime
import random

from pathlib import Path
from tqdm import tqdm

from typing import Any, Callable, Iterable, TypeVar, Generic, Sequence, List, Optional, Union

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Sampler

from tensorboardX import SummaryWriter


import method
from utils import lr_protocols, criterion
from utils.utils import earlystop, seg_metrics
from utils.datasets import dataloader




def _logging(args, writer, phase, epoch, score, loss, verbose = False):
    """Write tensorboardX.SummaryWriter and print results.
    """
    writer.add_scalar(f'IoU BG/{phase}', score['Class IoU'][0], epoch)
    writer.add_scalar(f'IoU Nerve/{phase}', score['Class IoU'][1], epoch)
    writer.add_scalar(f'Dice BG/{phase}', score['Class Dice'][0], epoch)
    writer.add_scalar(f'Dice Nerve/{phase}', score['Class Dice'][1], epoch)
    writer.add_scalar(f'Precision BG/{phase}', score['Class Precision'][0], epoch)
    writer.add_scalar(f'Precision Nerve/{phase}', score['Class Precision'][1], epoch)
    writer.add_scalar(f'Recall BG/{phase}', score['Class Recall'][0], epoch)
    writer.add_scalar(f'Recall Nerve/{phase}', score['Class Recall'][1], epoch)
    writer.add_scalar(f'Specificity BG/{phase}', score['Class Specificity'][0], epoch)
    writer.add_scalar(f'Specificity Nerve/{phase}', score['Class Specificity'][1], epoch)
    writer.add_scalar(f'epoch loss/{phase}', loss, epoch)

    if verbose:
        print("[{}] Epoch: {}/{} Loss: {:.5f}".format(phase, epoch, args.total_itrs, loss))
        print("\tDice [0]: {:.5f} [1]: {:.5f}".format(score['Class Dice'][0], score['Class Dice'][1]))
        print("\tIoU [0]: {:.5f} [1]: {:.5f}".format(score['Class IoU'][0], score['Class IoU'][1]))
        print("\tPrecision: {:.2f}, Recall: {:.2f}, Specificity: {:.2f}".format(score['Class Precision'][1], score['Class Recall'][1], score['Class Specificity'][1]))

def get_argparser():
    parser = argparse.ArgumentParser()

    # Experiment option
    parser.add_argument("--run_test", action="store_true", 
                        help="  (default: False)")
    parser.add_argument("--demo", action="store_true",
                        help=" (default: False)")
    parser.add_argument("--short_memo", default="short memo", 
                        help="")
    parser.add_argument("--login", type=str, default="/home/dongik/src/login.json", 
                        help="SMTP info. (login: ID&PW)")
    parser.add_argument("--version", default=None, 
                        help="")
    parser.add_argument("--folder", type=str, default="/home/dongik/src/CINblocks/.cache", 
                        help="Path to the log folder")

    # Datatset [torch.utils.data.DataLoader]
    parser.add_argument("--train_batch_size", type=int, default=32,
                        help="[train] how many samples per batch to load.")
    parser.add_argument("--val_batch_size", type=int, default=16,
                        help="[validate] how many samples per batch to load.")
    parser.add_argument("--test_batch_size", type=int, default=16,
                        help="[test] how many samples per batch to load.")
    parser.add_argument("--shuffle", action="store_false",
                        help="Set to True to have the data reshuffled at every epoch. (default: True)")
    parser.add_argument("--sampler", default=None,
                        help="Defines the strategy to draw samples from the dataset.")
    parser.add_argument("--batch_sampler", default=None,
                        help="Like sampler, but returns a batch of indices at a time.")
    parser.add_argument("--num_workers", type=int, default=8, 
                        help="How many subprocesses to use for data loading.\
                            0 means that the data will be loaded in the main process. (default: 8)")
    parser.add_argument("--drop_last", action="store_true",
                        help="Set to True to drop the last incomplete batch,\
                            if the dataset size is not divisible by the batch size. (default: False)")
    # Dataset [common arguments]
    parser.add_argument("--dataset_pth_prefix", type=str, default="/home/dongik/datasets",
                        help="")
    parser.add_argument("--dataset", choices=["BUSI_with_GT", "Allnerve"], default="BUSI_with_GT",
                        help="")
    parser.add_argument("--kfold", type=int, default=4,
                        help="kfold (default: 4)")
    parser.add_argument("--k", type=int, default=0, 
                        help="i-th fold set of kfold data (default: 0)")
    parser.add_argument("--resize", nargs='+', default=[224, 224],
                        help="")
    parser.add_argument("--padding", type=bool, default=False,
                        help="")
    parser.add_argument("--padding_size", nargs='+', default=[1024, 1024],
                        help="")
    parser.add_argument("--scale", type=float, default=0.25,
                        help="")
    
    # Train Dataset [BUSI & Allnerve]
    parser.add_argument("--trainset", default={
        'median' : {
            'region' : ['forearm', 'wrist'],
            'modality' : ['HM', 'SN']
        },
        'peroneal' : {
            'region' : ['FH', 'FN', 'FN+1', 'FN+2', 'FN+3', 'FN+4'],
            'modality' : ['UN']
        },
        'BUSI_with_GT' : {
            'category' : ["benign", "malignant", "normal"]
        }
    }, help="")
    # Test Dataset [BUSI & Allnerve]
    parser.add_argument("--testset", default={
        'median' : {
            'region' : ['forearm', 'wrist'],
            'modality' : ['HM', 'SN']
        },
        'peroneal' : {
            'region' : ['FH', 'FN', 'FN+1', 'FN+2', 'FN+3', 'FN+4'],
            'modality' : ['UN']
        },
        'BUSI_with_GT' : {
            'category' : ["benign", "malignant", "normal"]
        }
    }, help="")

    # Model option
    available_models = sorted(
        name for name in method.method.__dict__ if name.islower() and \
        not (name.startswith("__") or name.startswith('_')) and callable( method.method.__dict__[name]) 
    )

    # Model [...]
    parser.add_argument("--model", type=str, default="_unet", choices=available_models,
                        help="Model name (default: _unet)")
    
    # Model [Conditional Adaptation]
    parser.add_argument("--reduction_ratio", type=int, default=16,
                        help="")
    parser.add_argument("--no_att", action='store_false', 
                        help="")
    parser.add_argument("--no_cin", action='store_false',
                        help="")
    
    # Model [CINB]
    parser.add_argument("--emb_classes", type=int, default=2, 
                        help="Embedding classes (default: 2)")
    parser.add_argument("--IN_affine", action='store_true', 
                        help="")
    parser.add_argument("--CIN_affine", action='store_false',
                        help="")

    # Training option
    parser.add_argument("--exp_itrs", type=int, default=1, 
                        help="Repeat n identical experiments (default: 1)")
    parser.add_argument("--random_seed", type=int, default=1, 
                        help="Random seed (default: 1)")
    parser.add_argument("--total_itrs", type=int, default=200,
                        help="Total iteration epoch number (default: 0.2k)")
    parser.add_argument("--verbose", action='store_true',
                        help="")
    

    # Criterion option
    parser.add_argument("--loss_type", type=str, default="entropydice",
                        help="Defines loss function (default: entropydice)")
    

    # Learning protocols option
    parser.add_argument("--optimizer", type=str, default='sgd',
                        help="optimizer (default: sgd)")
    parser.add_argument("--scheduler", type=str, default='steplr',
                        help="scheduler (default: steplr)")
    # Optimizer [SGD]
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="learning rate (default: 1e-3)")
    parser.add_argument("--momentum", type=float, default=0.9,
                        help='momentum (default: 0.9)')
    parser.add_argument("--weight_decay", type=float, default=5e-4,
                        help='weight decay (default: 5e-4)')
    
    # Scheduler [StepLR]
    parser.add_argument("--step_size", type=int, default=100, 
                        help="step size (default: 100)")
    parser.add_argument("--gamma", type=float, default=0.1,
                        help='weight decay (default: 0.1)')


    
    # Metric & early stop option
    parser.add_argument("--patience", type=int, default=50,
                        help="Number of epochs with no improvement after which training will be stopped (default: 50)")
    parser.add_argument("--delta", type=float, default=0.001,
                        help="Minimum change in the monitored quantity to qualify as an improvement (default: 0.001)")



    return parser.parse_args()

def run(args, RUN, FOLD) -> dict:

    # Experiment setting
    torch.manual_seed(
        seed=args.random_seed
    )
    np.random.seed(
        seed=args.random_seed
    )
    random.seed(args.random_seed)

    devices = torch.device(
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    )

    # DataLoader (torch.utils.data.DataLoader)
    loader = dataloader.get_loader(
        args=args,
        verbose=True
    )

    # Model (nn.Module)
    model = method.method.__dict__[args.model](
        args=args
    )
    if torch.cuda.device_count() > 1:
        print("cuda multiple GPUs")
        model = nn.DataParallel(model)
    else:
        print("cuda single GPUs")
    model.to(devices)

    # Optimizer (torch.optim.Optimizer)
    param = [
        {
            "params" : model.parameters()
        }
    ]
    optimizer = lr_protocols.optimizers.__dict__[args.optimizer](
        param=param,
        args=args
    )
    scheduler = lr_protocols.schedulers.__dict__[args.scheduler](
        optimizer=optimizer,
        args=args
    )

    # Logging and Training tools
    writer = SummaryWriter(
        log_dir=os.path.join(
            args.folder,
            "tensorboard",
            RUN, 
            FOLD
        )
    )
    metrics = seg_metrics.StreamSegMetrics(
        n_classes=2
    )
    early_stop = earlystop.EarlyStopping(
        patience=args.patience,
        delta=args.delta, 
        verbose=args.verbose, 
        path=os.path.join(
            args.folder,
            "best-param",
            RUN, 
            FOLD
        ),
        ceiling=False,
    )

    # Criterion
    loss = criterion.criterion.__dict__[args.loss_type](
        args=args
    )

    # Train/Validate
    start_time = datetime.datetime.now()
    bscore = {}
    for epoch in range(0, args.total_itrs):
        
        # Train
        model.train()
        metrics.reset()
        running_loss = 0.0

        pbar = tqdm(
            enumerate(loader["train"]), 
            total=len(loader["train"]),
            desc=f"Train {epoch}/{args.total_itrs}",
            ncols=70,
            ascii=" =",
            leave=False)
        
        for i, sample in pbar:
            pbar.set_description(f"Train {epoch}/{args.total_itrs}")

            optimizer.zero_grad()

            for k, v in sample.items():
                sample[k] = sample[k].to(devices)

            ims = sample["image"].to(devices)
            lbls = sample["mask"].to(devices)
            bbox = sample["coordinates"].to(devices)
            cls = sample["cls"].to(devices)
            
            out = model(sample)
            output = loss(out, sample)
            output.backward()

            probs = nn.Softmax(dim=1)(out['image'])
            preds = torch.max(probs, 1)[1]

            metrics.update(
                label_trues=lbls.detach().cpu().numpy(),
                label_preds=preds.detach().cpu().numpy()
            )
            running_loss += output.item() * ims.size(0)


            optimizer.step()
        
        scheduler.step()

        epoch_loss = running_loss / len(loader["train"])
        score = metrics.get_results()

        _logging(
            args=args,
            writer=writer,
            phase="train",
            epoch=epoch,
            score=score,
            loss=epoch_loss,
            verbose=args.verbose
        )


        # Validate
        model.eval()
        metrics.reset()
        running_loss = 0.0

        pbar = tqdm(
                enumerate(loader["val"]), total=len(loader["val"]), 
                desc=f"Test {epoch}/{args.total_itrs}",
                ncols=70,
                ascii=" =",
                leave=False)
        
        with torch.no_grad():
            for i, sample in pbar:
                pbar.set_description(f"Test {epoch}/{args.total_itrs}")
                
                for k, v in sample.items():
                    sample[k] = sample[k].to(devices)
                
                ims = sample["image"].to(devices)
                lbls = sample["mask"].to(devices)
                bbox = sample["coordinates"].to(devices)
                cls = sample["cls"].to(devices)
                
                out = model(sample)
                output = loss(out, sample)

                probs = nn.Softmax(dim=1)(out['image'])
                preds = torch.max(probs, 1)[1]

                metrics.update(
                    label_trues=lbls.detach().cpu().numpy(),
                    label_preds=preds.detach().cpu().numpy()
                )
                running_loss += output.item() * ims.size(0)

            epoch_loss = running_loss / len(loader["train"])
            score = metrics.get_results()
        
        _logging(
            args=args,
            writer=writer,
            phase="val",
            epoch=epoch,
            score=score,
            loss=epoch_loss,
            verbose=args.verbose
        )

        #print("Epoch loss", epoch_loss)
        if early_stop(score=epoch_loss, model=model, epoch=epoch):
            bscore = score
            bloss = epoch_loss
            
        if early_stop.early_stop:
            print("Early Stop !!!")
            break
        
        if args.demo and epoch == 0:
            print("Run Demo !!!")
            break


    return {
        "Precision" : {
            "background" : bscore['Class Precision'][0],
            "RoI" : bscore['Class Precision'][1]
        },
        "Recall" : {
            "background" : bscore['Class Recall'][0],
            "RoI" : bscore['Class Recall'][1]
        },
        "Specificity" : {
            "background" : bscore['Class Specificity'][0],
            "RoI" : bscore['Class Specificity'][1]
        },
        "IoU" : {
            "background" : bscore['Class IoU'][0],
            "RoI" : bscore['Class IoU'][1]
        },
        "Dice score" : {
            "background" : bscore['Class Dice'][0],
            "RoI" : bscore['Class Dice'][1]
        },
        "time elapsed" : str(datetime.datetime.now() - start_time)
    }



if __name__ == "__main__":
    
    from utils.datasets.allnerve import Allnerve
    from torch.utils.data import DataLoader

    args = argparse.ArgumentParser()
    args.add_argument("--padding", action='store_true', default=False)
    args.add_argument("--show", action='store_true', default=False)
    args = args.parse_args()

    image_set_type = ['train', 'val']

    for ist in image_set_type:
        dst = Allnerve(
            dataset_pth_prefix="/home/dongik/datasets",

            fold="v5/1",
            image_set=ist,
            padding=args.padding,
            padding_size=(1024, 1024),
            show=args.show
        )
        loader = DataLoader(
            dst, 
            batch_size=16, 
            shuffle=True, 
            num_workers=1, 
            drop_last=False
        )
        print(f'len [{ist}]: {len(dst)}')

        for i, sample in tqdm(enumerate(loader), total=len(loader)):
            if i == 0:
                print(type(sample))
                print(sample['cls'].size(), sample['cls'])
                print(sample['center'].size())
                print(sample['image'].size(), sample['image'].dtype)
                print(sample['mask'].size(), sample['mask'].dtype)
        print('Clear !!!')



    args = get_argparser()

    for k, v in vars(args).items():
        print(f"{k}: {v}, type: {type(v)}")
        if isinstance(v, Path):
            print(f"{k} is Path: {v.is_dir()}")
