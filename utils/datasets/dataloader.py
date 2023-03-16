from torch.utils.data import DataLoader

from .busi import Busi
from .allnerve import Allnerve


def get_loader(args, verbose=True) -> dict:

    loader = {}

    if args.run_test:
        if args.dataset == "BUSI_with_GT":
            test_dst = Busi(
                dataset_pth_prefix = args.dataset_pth_prefix,
                
                testset = args.testset,
                fold = f"v{args.kfold}/{args.k}",
                image_set = "val",
                padding = args.padding,
                padding_size = args.padding_size,
                scale = args.scale,
                resize = args.resize
            )
        elif args.dataset == "Allnerve":
            test_dst = Allnerve(
                dataset_pth_prefix = args.dataset_pth_prefix,
                
                testset = args.testset,
                fold = f"v{args.kfold}/{args.k}",
                image_set = "val",
                padding = args.padding,
                padding_size = args.padding_size,
                scale = args.scale,
                resize = args.resize
            )
        else:
            raise Exception( f'{args.dataset}: Not supported' )

        loader["test"] = DataLoader(
            dataset = test_dst,
            batch_size = args.test_batch_size,
            shuffle = args.shuffle,
            sampler = args.sampler,
            batch_sampler = args.batch_sampler,
            num_workers = args.num_workers, 
            drop_last = args.drop_last
        )
        
        if verbose:
            print(f'Dataset: {args.dataset}, v{args.kfold}/{args.k}')
            print(f'\t[Test]: {len(test_dst)}')
            if args.padding:
                print(f'\tPadding size: {args.padding_size}, Scale: {args.scale}')
            else:
                print(f'\tResize: {args.resize}')
            print(f'\tShuffle: {args.shuffle}')
            print(f'\tNum workers: {args.num_workers}')
            print(f'\tDrop last: {args.drop_last}')

    else:
        if args.dataset == 'BUSI_with_GT':
            train_dst = Busi(
                dataset_pth_prefix=args.dataset_pth_prefix,
                
                trainset = args.trainset,
                testset=args.testset,
                fold=f'v{args.kfold}/{args.k}',
                image_set='train',
                padding=args.padding,
                padding_size=args.padding_size,
                scale=args.scale,
                resize=args.resize
            )
            val_dst = Busi(
                dataset_pth_prefix=args.dataset_pth_prefix,
                
                trainset = args.trainset,
                testset=args.testset,
                fold=f'v{args.kfold}/{args.k}',
                image_set='val',
                padding=args.padding,
                padding_size=args.padding_size,
                scale=args.scale,
                resize=args.resize
            )
        elif args.dataset == "Allnerve":
            train_dst = Allnerve(
                dataset_pth_prefix = args.dataset_pth_prefix,
                trainset = args.trainset,
                testset = args.testset,
                fold = f"v{args.kfold}/{args.k}",
                image_set = "train",
                padding = args.padding,
                padding_size = args.padding_size,
                scale = args.scale,
                resize = args.resize
            )
            val_dst = Allnerve(
                dataset_pth_prefix = args.dataset_pth_prefix,
                trainset = args.trainset,
                testset = args.testset,
                fold = f"v{args.kfold}/{args.k}",
                image_set = "val",
                padding = args.padding,
                padding_size = args.padding_size,
                scale = args.scale,
                resize = args.resize
            )
        else:
            raise Exception( args.dataset, ': Not supported' )
        
        loader["train"] = DataLoader(
            dataset = train_dst,
            batch_size = args.train_batch_size,
            shuffle = args.shuffle,
            sampler = args.sampler,
            batch_sampler = args.batch_sampler,
            num_workers = args.num_workers, 
            drop_last = args.drop_last
        )
        loader["val"] = DataLoader(
            dataset = val_dst,
            batch_size = args.val_batch_size,
            shuffle = args.shuffle,
            sampler = args.sampler,
            batch_sampler = args.batch_sampler,
            num_workers = args.num_workers, 
            drop_last = args.drop_last
        )
        
        if verbose:
            print(f'Dataset: {args.dataset}, v{args.kfold}/{args.k}')
            print(f'\t[Train]: {len(train_dst)}')
            print(f'\t[val]: {len(val_dst)}')
            if args.padding:
                print(f'\tPadding size: {args.padding_size}, Scale: {args.scale}')
            else:
                print(f'\tPadding: {args.padding}, Resize: {args.resize}')
            print(f'\tShuffle: {args.shuffle}')
            print(f'\tNum workers: {args.num_workers}')
            print(f'\tDrop last: {args.drop_last}')
        
    return loader



if __name__ == "__main__":
    pass