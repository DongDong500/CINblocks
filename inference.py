import os
import argparse
import random

import numpy as np

from tqdm import tqdm

import torch
import torch.nn as nn

import method
from utils.utils import utils, seg_metrics
from utils.datasets import dataloader

def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_pth", type=str, default=None)
    parser.add_argument("--exp", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=0)
    parser.add_argument("--CIN_affine", action='store_false')

    return parser.parse_args()

def _prst(score):
    print("\tDice [0]: {:.5f} [1]: {:.5f}".format(score['Class Dice'][0], score['Class Dice'][1]))
    print("\tIoU [0]: {:.5f} [1]: {:.5f}".format(score['Class IoU'][0], score['Class IoU'][1]))
    print("\tPrecision: {:.2f}, Recall: {:.2f}, Specificity: {:.2f}".format(score['Class Precision'][1], score['Class Recall'][1], score['Class Specificity'][1]))



if __name__ == "__main__":
    opts = get_argparser()

    if isinstance(opts.json_pth, str):
        opts.json_pth = [opts.json_pth]
    else:
        opts.json_pth = [
            "Feb22-00-30-37",
            "Feb21-17-30-55",
            "Feb21-08-55-08",
            "Feb21-21-52-59",
            "Feb21-13-45-39",
            "Feb21-04-52-29"
        ]
    
    idx = 0
    for jpth in opts.json_pth:
        idx += 1
        
        args = utils.Params(
            json_path=os.path.join(
                "/home/dongik/src/USESnet-result",
                jpth,
                "param-summary.json"
            )
        )
        args._update(
            {
            "CIN_affine" : opts.CIN_affine
            }
        )

        # Experiment setting
        torch.manual_seed(
            seed=args.random_seed
        )
        np.random.seed(
            seed=args.random_seed
        )
        random.seed(args.random_seed)

        args.run_test = True

        if opts.batch_size == 0:
            args.test_batch_size = args.val_batch_size
        else:
            args.test_batch_size = opts.batch_size

        devices = torch.device(
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        )
        metrics = seg_metrics.StreamSegMetrics(
            n_classes=2
        )
        
        ts = [
            {
            'BUSI_with_GT' : {
            'category' : ["benign"]
            }
            },
            {
            'BUSI_with_GT' : {
            'category' : ["malignant"]
            }
            },
        ]

        with open(
                file=os.path.join(
                    "/home/dongik/src/CINblocks",
                    "result-busi.txt"
                ),
                mode="a"
            ) as f:
                f.write(f"[{idx}]\n")
                f.write(f"{args.short_memo}\nFolder: {jpth}\n")
                f.write(f"Model: {args.model}\n")

        for tset in ts:
            
            args.testset = tset
            arr = {}

            for i in range(args.kfold):

                # DataLoader (torch.utils.data.DataLoader)
                args.k = i
                loader = dataloader.get_loader(
                    args=args,
                    verbose=True
                )

                # Load model & params
                resume_ckpt = os.path.join(
                    args.folder,
                    "best-param",
                    f"run_0{opts.exp}/fold_0{i}",
                    "checkpoint.pt"
                )
                if not os.path.exists(resume_ckpt):
                    raise FileNotFoundError(resume_ckpt)
                ckpt = torch.load(
                    f=resume_ckpt,
                    map_location='cpu'
                )
                print(f"[run_0{opts.exp}/fold_0{i}] Best epoch: {ckpt['epoch']}")

                # Model (nn.Module)
                model = method.method.__dict__[args.model](
                    args=args
                )
                if torch.cuda.device_count() > 1:
                    print("cuda multiple GPUs")
                    model = nn.DataParallel(model)
                else:
                    print("cuda single GPUs")
                model.load_state_dict(ckpt["model_state"])
                model.to(devices)

                model.eval()
                metrics.reset()
                with torch.no_grad():
                    for i, sample in tqdm(
                        enumerate(loader["test"]), 
                        total=len(loader["test"]),
                        desc="Test",
                        ncols=70,
                        ascii=" =",
                        leave=False
                    ):
                        ims = sample["image"].to(devices)
                        lbls = sample["mask"].to(devices)
                        bbox = sample["coordinates"].to(devices)
                        cls = sample["cls"].to(devices)
                        
                        out = model(sample)

                        probs = nn.Softmax(dim=1)(out['image'])
                        preds = torch.max(probs, 1)[1]

                        metrics.update(
                            label_trues=lbls.detach().cpu().numpy(),
                            label_preds=preds.detach().cpu().numpy()
                        )
                    score = metrics.get_results()
                #_prst(score)

                if not bool(arr):
                    arr = {}
                    for k, val in score.items():
                        arr[k] = [val[1]]
                else:
                    for k, val in score.items():
                        arr[k].append(val[1])

            with open(
                file=os.path.join(
                    "/home/dongik/src/CINblocks",
                    "result-busi.txt"
                ),
                mode="a"
            ) as f:
                f.write(f"Test set: {tset}\n")
                for k, val in arr.items():
                    r = np.array(val)
                    print(f"[{k}]\n\t{r.mean():.4f}±{r.std():.4f}")
                    f.write(f"[{k}]\n\t{r.mean():.4f}±{r.std():.4f}\n")
        