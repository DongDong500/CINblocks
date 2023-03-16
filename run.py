
import os
import datetime
import traceback
import numpy as np

from main import get_argparser, run
from utils.utils import email, utils




def avg_rst(obj):
    """average results
    Return
    {
        "Dice score" : {
            "mean" : np.array(result[key]).mean(),
            "std" : np.array(result[key]).std()
        },
        "IoU" : {},
        "Precision" : {},
        "Recall" : {},
        "Specificity" : {}
    }
    """

    result = {
        "Precision" : [],
        "Recall" : [],
        "Specificity" : [],
        "IoU" : [],
        "Dice score" : []
    }
    rsts = {}
    
    for key, rst in result.items():
        for fold, v in obj.items():
            result[key].append(v[key]["RoI"])
        
        rsts[key] = {
            "mean" : np.array(result[key]).mean(),
            "std" : np.array(result[key]).std()
        }

    return rsts

def _run(arg, flag = 0) -> dict:
    """Run main.py with the given arguments.
    """
    results = {}

    for exp_itrs in range(0, args.exp_itrs):
        print(f"---- Iteration ({exp_itrs + 1}) ----")
        start_time = datetime.datetime.now()

        RUN = 'run_' + str(exp_itrs).zfill(2)
        results[RUN] = {}
        for i in range(0, args.kfold):
            FOLD = 'fold_' + str(i).zfill(2)
            args.k = i

            print(f" ({i + 1}) Fold")

            os.makedirs(
                os.path.join(
                    args.folder,
                    "tensorboard",
                    RUN,
                    FOLD
                )
            )
            os.makedirs(
                os.path.join(
                    args.folder,
                    "best-param",
                    RUN,
                    FOLD
                )
            )
            results[RUN][FOLD] = run(args, RUN, FOLD)

        result = avg_rst(results[RUN])

        Rsts = ""
        for k, v in result.items():
            Rsts += f"{k}: \n{v['mean']:.4f}\n±{v['std']:.4f}\n"

        Tts = ""
        if args.dataset == "BUSI_with_GT":
            dataset_str = f"Dataset: BUSI_with_GT/"
            Tts = f"train with {args.trainset['BUSI_with_GT']['category']}, test with {args.testset['BUSI_with_GT']['category']}"
        elif args.dataset == "Allnerve":
            dataset_str = f"Dataset: Allnerve/"
            Tts = f"train with {args.trainset}, test with {args.testset}"
        else:
            raise Exception

        email.Email(
            subject = f"[{flag}] {args.model} / {args.short_memo}",
            msg = {
                "Short Memo" : f"Description: [{RUN}] " + args.short_memo,
                "Description" : f"Iteration: {exp_itrs + 1}/{args.exp_itrs}, K-fold (K={args.kfold}) summary",
                "Folder" : "Folder: " + args.version,
                "Dataset" : dataset_str,
                "DataTransform" : f"Padding size: {args.padding_size}, Scale: {args.scale}" if args.padding else f"Padding: {args.padding}, Resize: {args.resize}",
                "Train Tatics" : Tts,
                "Model" : f"Model : {args.model}",
                "Summarize results" : Rsts,
                "Time elapsed" : "Time elapsed: " + str(datetime.datetime.now() - start_time)
            }
        ).send()

    return results



if __name__ == "__main__":
    
    total_time = datetime.datetime.now()

    args = get_argparser()

    try:
        is_error = False    
        ### Ablation Study Zone ###

        flag = 0

        args.dataset = "BUSI_with_GT"
        args.resize = (256, 256)
        args.emb_classes = 3

        # 총 6 + 12 개의 실험.
        
        mode = {
            "as-a" : {
                "model" : [
                    '_unet_cin_slim'
                ],
                "itr" : {
                    "shortmemo" : [
                        '(CIN) with cin-affine BUSI / Rsz256-ElyStpLoss / benign + malignant',
                        '(CIN) with cin-affine BUSI / Rsz256-ElyStpLoss / benign + malignant + normal'
                    ],
                    "trainset" : [
                        {
                            'BUSI_with_GT' : {
                                'category' : ['benign', 'malignant']
                            }
                        },
                        {
                            'BUSI_with_GT' : {
                                'category' : ['benign', 'malignant', 'normal']
                            }
                        },
                    ],
                    "testset" : [
                        {
                            'BUSI_with_GT' : {
                                'category' : ['benign', 'malignant']
                            }
                        },
                        {
                            'BUSI_with_GT' : {
                                'category' : ['benign', 'malignant', 'normal']
                            }
                        },
                    ]
                }
            },
            "as-b" : {
                "model" : [
                    '_unet_cin_slim'
                ],
                "itr" : {
                    "shortmemo" : [
                        '(CIN) without cin-affine BUSI / Rsz256-ElyStpLoss / benign + malignant',
                        '(CIN) without cin-affine BUSI / Rsz256-ElyStpLoss / benign + malignant + normal'
                    ],
                    "trainset" : [
                        {
                            'BUSI_with_GT' : {
                                'category' : ['benign', 'malignant']
                            }
                        },
                        {
                            'BUSI_with_GT' : {
                                'category' : ['benign', 'malignant', 'normal']
                            }
                        },
                    ],
                    "testset" : [
                        {
                            'BUSI_with_GT' : {
                                'category' : ['benign', 'malignant']
                            }
                        },
                        {
                            'BUSI_with_GT' : {
                                'category' : ['benign', 'malignant', 'normal']
                            }
                        },
                    ]
                }
            },
            "pp" :{
                "model" : [
                    '_unet3plus_cin',
                    '_unet2plus_cin',
                    '_unet_cin',
                ],
                "itr" : {
                    "shortmemo" : [
                        '(CIN) without cin-affine BUSI / Rsz256-ElyStpLoss / benign + malignant',
                        '(CIN) without cin-affine BUSI / Rsz256-ElyStpLoss / benign + malignant + normal'
                    ],
                    "trainset" : [
                        {
                            'BUSI_with_GT' : {
                                'category' : ['benign', 'malignant']
                            }
                        },
                        {
                            'BUSI_with_GT' : {
                                'category' : ['benign', 'malignant', 'normal']
                            }
                        },
                    ],
                    "testset" : [
                        {
                            'BUSI_with_GT' : {
                                'category' : ['benign', 'malignant']
                            }
                        },
                        {
                            'BUSI_with_GT' : {
                                'category' : ['benign', 'malignant', 'normal']
                            }
                        },
                    ]
                }
            }
        }

        ### ------------------- ###
        for key, _ in mode.items():
            
            if key == 'as-a':
                args.CIN_affine = True
            else:
                args.CIN_affine = False

            model = mode[key]['model']

            shortmemo = mode[key]['itr']['shortmemo']
            category = mode[key]['itr']['trainset']
            testset = mode[key]['itr']['testset']

            for mdl in model:
                args.model = mdl
                for srtm, ctgy, tset in zip(shortmemo, category, testset):
                    flag += 1
                    args.short_memo = srtm
                    args.trainset = ctgy
                    args.testset = tset

                    current_working_dir = os.path.dirname(os.path.abspath(__file__))
                    current_time = datetime.datetime.now().strftime('%b%d-%H-%M-%S')
                    
                    if args.demo:
                        current_time += "_demo"
                    
                    args.version = current_time
                    args.folder = os.path.join(
                        current_working_dir + "-result",
                        current_time
                    )
                    utils.mklogdir(
                        folder=args.folder, 
                        version=args.version, 
                        verbose=True
                    )
                    utils.save_argparser(
                        parser = args,
                        save_dir = args.folder
                    )
                    utils.save_dict_to_json(
                        d = _run(arg=args, flag=flag),
                        json_path = os.path.join(
                            args.folder, 
                            "result.json"
                        )
                    )

    except KeyboardInterrupt:
        is_error = True
        print("KeyboardInterrupt: Stop !!!")

    except Exception as e:
        is_error = True
        print("Error:", e)
        print(traceback.format_exc())

    try:
        if is_error:
            os.rename(args.folder, args.folder+"_aborted")
    except FileNotFoundError as e:
        print("File does not exists.")
        print(f"{args.folder}: {os.path.exists(args.folder)}")
        
    total_time = datetime.datetime.now() - total_time
    print(f"Time elapsed (h:m:s.ms) {total_time}") 
