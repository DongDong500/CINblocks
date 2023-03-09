import os
from typing import Tuple, Optional, Iterable, List

import numpy as np
import torch.utils.data as data
from PIL import Image
 
from .custom_transforms import Compose, Scale, Resize, Pad, ToTensor, Normalize
from .utils import pad, show_sample

# https://doi.org/10.1016/j.dib.2019.104863

C = {
    'benign' : 0,
    'malignant' : 1,
    'normal' : 2
}

class Busi(data.Dataset):
    """Dataset (BUSI_with_GT) description: 
    benign          (437 samples)
    malignant       (210 samples)
    normal          (133 samples)
    splits
        benign
            v5      0/train.txt or 0/val.txt
        malignant
            v5      0/train.txt or 0/val.txt
        normal
            v5      0/train.txt or 0/val.txt
    """
    def __init__(self, 
                dataset_pth_prefix: str = '/home/dongik/datasets', 
                
                trainset: dict = {
                    'BUSI_with_GT' : {
                        'category' : ["benign", "malignant", "normal"]
                    }
                },
                testset: dict = {
                    'BUSI_with_GT' : {
                        'category' : ["benign", "malignant", "normal"]
                    }
                },

                fold: str = 'v5/3', 
                image_set: str = 'train', 

                padding: bool = False,
                padding_size: Tuple[int, int] = (1024, 1024),
                scale: float = 0.5,
                resize: List[int] = [224, 224],

                show: Optional[bool] = False,
                **kwargs):
        """
        Args:
            dataset_pth_prefix (str): Root directory path of the ultrasound 
                peripheral nerve dataset. (default: ``/home/dongik/datasets``)
            
            fold (str):  Path to the i-th sample list to use for k-fold 
                cross-validation. (E.g. ``v5/3``, ``v4/2`` ... )
            image_set (str): Select the image_set to use. (E.g. ``train`` or ``val``)
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

        self.busi = []

        if self.image_set == 'train':
            self.ds = trainset
        elif self.image_set == 'val':
            self.ds = testset
        else:
            raise Exception
        
        if not isinstance(self.ds, dict):
            raise Exception
        
        for d, v in self.ds.items():
            for c in v['category']:
                self._category = c

                self.image_dir = os.path.join(
                    self.dataset_pth_prefix, 
                    d, 
                    self._category
                )
                self.split_file = os.path.join(
                    self.dataset_pth_prefix, 
                    d, 
                    'splits', 
                    self._category, 
                    fold, 
                    image_set + '.txt'
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
                    
                    if not os.path.exists( os.path.join(self.image_dir, f"{self._category} ({index}).png") ):
                        raise FileNotFoundError(os.path.join(self.image_dir, f"{self._category} ({index}).png"))
                    
                    if not os.path.exists( os.path.join(self.image_dir, f"{self._category} ({index})_mask.png") ):
                        raise FileNotFoundError(os.path.join(self.image_dir, f"{self._category} ({index})_mask.png") )
                    
                    img = Image.open( os.path.join(self.image_dir, f"{self._category} ({index}).png") ).convert('RGB')
                    lbl = Image.open( os.path.join(self.image_dir, f"{self._category} ({index})_mask.png") ).convert('L')
                    
                    assert( img.size == lbl.size )
            
                    if self.padding:
                        img, lbl = pad(img, lbl, self.padding_size)

                    if self._category == 'normal':
                        h, w = np.array([0, 0])
                    else:
                        h, w = np.where(np.array(lbl, dtype=np.uint8) > 0)

                    _busi = {
                        'image' : img,
                        'mask' : lbl,
                        'coordinates' : np.array([w.min(), h.min(), w.max()-w.min()+1, h.max()-h.min()+1], dtype=np.float32),
                        'center' : np.array([(h.max()+h.min()) / 2, (w.max()+w.min()) / 2], dtype=np.float32), # [H x W]
                        'height' : img.size[1],
                        'width' : img.size[0],
                        'category' : self._category,
                        'id' : int(index),
                        'cls' : C[self._category]
                    } 
                    self.busi.append( _busi )
                

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            sample (dict): {
                'image' : (Tensor),
                'mask' : (Tensor),
                'coordinates' : (Tensor) [x, y, bbox_w, bbox_h]
            }
        """
        if self.image_set == "train":
            sample = self.transform_tr(self.busi[index])
        elif self.image_set == 'val':
            sample = self.transform_val(self.busi[index])
        else:
            raise Exception
        
        if self.show:
            show_sample(busi=self.busi[index], sample=sample, verbose=True)
        
        return sample

    def __len__(self):
        return len(self.busi)
    
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
    