import os
from typing import Dict, Tuple, Iterable, Optional, Union, List

import numpy as np
import torch.utils.data as data
from PIL import Image
 
from .custom_transforms import Compose, Scale, Resize, Pad, ToTensor, Normalize
from .utils import pad, show_sample

MEDIANS = {
    "forearm" : {
        "HM" : {
            "str" : 641,
            "end" : 802,
        },
        "SN" : {
            "str" : 1114,
            "end" : 1304,
        }
    },
    "wrist" : {
        "HM" : {
            "str" : 0,
            "end" : 640,
        },
        "SN" : {
            "str" : 803,
            "end" : 1113,
        }
    }
}

PERONEAL = {
    "FH" : 91,
    "FN" : 106,
    "FN+1" : 77,
    "FN+2" : 58,
    "FN+3" : 49,
    "FN+4" : 29,
    "FN+5" : 16
}

C = {
    "peroneal" : {
        "FH" : 0,
        "FN" : 1,
        "FN+1" : 2,
        "FN+2" : 3,
        "FN+3" : 4,
        "FN+4" : 5,
    },
    "median" : {
        "wrist" : 6,
        "forearm" : 7
    }
}

class Allnerve(data.Dataset):
    """Dataset (All nerves, median+peroneal) description: 
    peroneal
        FN
            UN  ()
        FH+1
            UN  ()
        splits
            FH
                UN
                    v5  0/train.txt or val.txt
            FN+1
                UN
                    v5  0/train.txt or val.txt
    median
        forearm
            HM  (641~802)
            SN  (1114~1304)
        wrist
            HM  (0~640)
            SN  (803~1113)
        splits
            forearm
                HM
                    v5  0/train.txt or 0/val.txt
                SN
                    v5  0/train.txt or 0/val.txt
            wrist
                HM
                    v5  0/train.txt or 0/val.txt
                SN
                    v5  0/train.txt or 0/val.txt
    """
    def __init__(self, 
                dataset_pth_prefix: str = '/home/dongik/datasets', 

                trainset: dict = {
                    'median' : {
                        'region' : ['forearm', 'wrist'],
                        'modality' : ['HM', 'SN']
                    },
                    'peroneal' : {
                        'region' : ['FH', 'FN', 'FN+1', 'FN+2', 'FN+3', 'FN+4'],
                        'modality' : ['UN']
                    },
                },
                testset: dict = {
                    'median' : {
                        'region' : ['forearm', 'wrist'],
                        'modality' : ['HM', 'SN']
                    },
                    'peroneal' : {
                        'region' : ['FH', 'FN', 'FN+1', 'FN+2', 'FN+3', 'FN+4'],
                        'modality' : ['UN']
                    }
                },

                fold: str = 'v5/3',
                image_set: str = 'train', 
                
                padding: bool = False,
                padding_size: Tuple[int, int] = (1024, 1024),
                scale: float = 0.5,
                resize: List[int] = [224, 224],
                
                show: Optional[bool] = False,
                **kwargs) -> None:
        """
        Args:
            dataset_pth_prefix (str): Root directory path of the ultrasound 
                peripheral nerve dataset. (default: ``/home/dongik/datasets``)

            trainset (dict):            
            testset (dict): 

            fold (str): Path to the i-th sample list to use for k-fold 
                cross-validation. (E.g. ``v5/3``, ``v4/2`` ... )
            image_set (str): Select the image_set to use. 
                (E.g. ``train`` or ``val``)
            padding (bool, optional): If true, all images are 0 padded and centered 
                with the identical size (E.g. 1024x1024), else, all images are 
                resized with given size (E.g. 224x224). Default is False.
            padding_size (Tuple[int, int], optional): Zero padding size. 
                Default is (1024x1024).
            scale (float, optional): Resize the input PIL Image to the given scale.
            resize (Tuple[int, int], optional): Resize the input PIL Image to 
                the given size.
            show (bool, optional): If true, the transformed and the original 
                images are showed through ``plt.show()``. Default is False.
        """
        super().__init__()
        self.dataset_pth_prefix = dataset_pth_prefix
        self.fold = fold
        self.image_set = image_set
        self.padding = padding
        self.padding_size = padding_size
        self.scale = scale
        self.resize = resize
        self.show = show
        
        self.nerve = []

        if self.image_set == 'train':
            self.ds = trainset
        elif self.image_set == 'val':
            self.ds = testset
        else:
            raise Exception
        
        if not isinstance(self.ds, dict):
            raise Exception

        for d, v in self.ds.items():
            for r in v['region']:
                for m in v['modality']:

                    self.image_dir = os.path.join(
                        dataset_pth_prefix, 
                        d, 
                        r, 
                        m
                    )
                    self.split_file = os.path.join(
                        dataset_pth_prefix, 
                        d, 
                        'splits', 
                        r, 
                        m, 
                        fold, 
                        self.image_set + '.txt'
                    )

                    if not os.path.exists(self.image_dir):
                        raise Exception(
                            'Dataset path is not found or corrupted. Image path:', self.image_dir
                        )
                    if not os.path.exists(self.split_file):
                        raise Exception(
                            'split file is not found or corrupted. Splits file path:', self.split_file
                        )
                    
                    with open(os.path.join(self.split_file), "r") as f:
                        file_names = [x.strip('\n') for x in f.readlines()]
                    
                    for index in file_names:
   
                        if d == 'median':

                            if not os.path.exists( os.path.join(self.image_dir, f"median ({index}).jpg") ):
                                raise FileNotFoundError(os.path.join(self.image_dir, f"median ({index}).jpg"))
                            
                            if not os.path.exists( os.path.join(self.image_dir, f"median ({index})_mask.jpg") ):
                                raise FileNotFoundError(os.path.join(self.image_dir, f"median ({index})_mask.jpg") )
                            
                            img = Image.open( os.path.join(self.image_dir, f"median ({index}).jpg") ).convert('RGB')
                            lbl = Image.open( os.path.join(self.image_dir, f"median ({index})_mask.jpg") ).convert('L')

                        elif d == 'peroneal':

                            if not os.path.exists( os.path.join(self.image_dir, f"peroneal ({index}).bmp") ):
                                raise FileNotFoundError(os.path.join(self.image_dir, f"peroneal ({index}).bmp"))
                            
                            if not os.path.exists( os.path.join(self.image_dir, f"peroneal ({index})_mask.bmp") ):
                                raise FileNotFoundError(os.path.join(self.image_dir, f"peroneal ({index})_mask.bmp") )
                            
                            img = Image.open( os.path.join(self.image_dir, f"peroneal ({index}).bmp") ).convert('RGB')
                            lbl = Image.open( os.path.join(self.image_dir, f"peroneal ({index})_mask.bmp") ).convert('L')

                        else:
                            raise Exception
                        
                        assert( img.size == lbl.size )
                        
                        if self.padding:
                            img, lbl = pad(img, lbl, self.padding_size)
                        
                        h, w = np.where(np.array(lbl, dtype=np.uint8) > 0)

                        _nerve = {
                            'image' : img,
                            'mask' : lbl,
                            'coordinates' : np.array([w.min(), h.min(), w.max()-w.min()+1, h.max()-h.min()+1], dtype=np.float32), # [W, H, Wlen, Hlen]
                            'center' : np.array([(h.max()+h.min()) / 2, (w.max()+w.min()) / 2], dtype=np.float32), # [H x W]
                            'height' : img.size[1],
                            'width' : img.size[0],
                            'category' : f'{d}/{r}/{m}', # legacy
                            'region' : r,
                            'modality' : m,
                            'id' : index,
                            'cls' : C[d][r] # New info
                        }
                        self.nerve.append( _nerve )                

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            sample (dict): {
                'image' : (Tensor),
                'mask' : (Tensor),
                'coordinates' : (Tensor) [x, y, bbox_w, bbox_h],
                'center' : (Tensor) [x, y]
                'cls' : (Tensor) 0 or 1
            }
        """
        if self.image_set == "train":
            sample = self.transform_tr(self.nerve[index])
        elif self.image_set == 'val':
            sample = self.transform_val(self.nerve[index])
        else:
            raise Exception
        
        if self.show:
            show_sample(busi=self.nerve[index], sample=sample, verbose=True)
        
        return sample

    def __len__(self):
        return len(self.nerve)
    
    def transform_tr(self, sample):
        transform = Compose([
            Scale(scale=self.scale) if self.padding else Resize(size=self.resize),
            ToTensor(),
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
        return transform(sample)

    def transform_val(self, sample):
        transform = Compose([
            Scale(scale=self.scale) if self.padding else Resize(size=self.resize),
            ToTensor(),
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
        return transform(sample)
    


if __name__ == "__main__":
    pass
