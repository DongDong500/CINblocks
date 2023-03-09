import numpy as np

import torch
import torchvision.transforms.functional as F

from PIL import Image
from PIL import ImageOps

from typing import Tuple, Optional, Iterable, List

_pil_interpolation_to_str = {
    F.InterpolationMode.NEAREST: 'torchvision.transforms.functional.InterpolationMode.NEAREST',
    F.InterpolationMode.BILINEAR : 'torchvision.transforms.functionalInterpolationMode.BILINEAR',
}

#
#  Custom Transforms for BUSI & All Nerve Semantic Segmentation
#

class Compose(object):
    """Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    Example:
        >>> transforms.Compose([
        >>>     transforms.Resize(size=(224, 224)),
        >>>     transforms.ToTensor(),
        >>> ])
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class Pad(object):
    """Pads the input PIL Image boundaries with zero
    Args:
        size (sequence): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If image size is larger 
            than the given size, raise Exception.
    """
    def __init__(self, size: Tuple[int, int]=(512, 512)):
        self.size = (size[1], size[0]) # (HxW) -> (WxH)
        self.left, self.right, self.top, self.bottom = 0, 0, 0, 0

    def __call__(self, sample):
        """
        Args:
            sample (dict): BUSI sample dictionary, Image to be padded.
        Returns:
            (dict): padded image.
        """
        image = sample['image'] # PIL.Image
        mask = sample['mask'] # PIL.Image
        coordinates = sample['coordinates'] # np.ndarray
        cls = sample['cls'] # int

        if self.size[0] < image.size[0] or self.size[1] < image.size[1]:
            raise Exception("Padding size must be smaller than the given image.\n", f"Padding size: {self.size}, image: {image.size}")

        # Padding
        width = self.size[0] - image.size[0]
        height = self.size[1] - image.size[1]

        if width % 2 == 0 and width > 0:
            self.left, self.right = width/2, width/2
        elif width % 2 == 1:
            self.left, self.right = (width + 1)/2, (width - 1)/2
        else:
            self.left, self.right = 0, 0

        if height % 2 == 0 and height > 0:
            self.top, self.bottom = height/2, height/2
        elif height % 2 == 1:
            self.top, self.bottom = (height + 1)/2, (height - 1)/2
        else:
            self.top, self.bottom = 0, 0
        
        border = (int(self.left), int(self.top), int(self.right), int(self.bottom))

        image = ImageOps.expand(image, border, fill=0)
        mask = ImageOps.expand(mask, border, fill=0)

        #add = np.array([self.left, self.top, 0, 0], dtype=np.float32)
        if image.size != self.size:
            raise Exception("Padding size does not match", image.size, self.size)

        return {
            'image' : image,
            'mask' : mask,
            'coordinates' : coordinates + np.array([self.left, self.top, 0, 0], dtype=np.float32),
            'cls' : cls
        }

    def __repr__(self) -> str:
        return self.__class__.__name__ + '(padding size {0}, [{1}, {2}, {3}, {4}])'.format(self.size, self.left, self.right, self.top, self.bottom)


class Resize(object):
    """Resize the input PIL Image to the given size.
    Args:
        size (sequence): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``F.InterpolationMode.BILINEAR``
    """
    def __init__(self, size: List[int], interpolation=F.InterpolationMode.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, sample):
        """
        Args:
            sample (dict): BUSI sample dictionary, Image to be scaled.
        Returns:
            (dict): Resized image.
        """
        image = sample['image']
        mask = sample['mask']
        coordinates = sample['coordinates']
        cls = sample['cls']
        center = sample['center']
        
        w_r = self.size[1] / sample['width']
        h_r = self.size[0] / sample['height']

        return {
            'image' : F.resize(image, self.size, self.interpolation),
            'mask' : F.resize(mask, self.size, self.interpolation),
            'coordinates' : coordinates * np.array([w_r, h_r, w_r, h_r], dtype=np.float32),
            'center' : center * np.array([h_r, w_r], dtype=np.float32),
            'cls' : cls
        }

    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        return self.__class__.__name__ + '(size={0}, interpolation={1})'.format(self.size, interpolate_str) 


class Scale(object):
    """Resize the input PIL Image to the given scale.
    Args:
        Scale (sequence or int): scale factors
        interpolation (int, optional): Desired interpolation. Default is
            ``F.InterpolationMode.BILINEAR``
    """
    def __init__(self, scale: float, interpolation=F.InterpolationMode.BILINEAR):
        self.scale = scale
        self.interpolation = interpolation

    def __call__(self, sample):
        """
        Args:
            sample (dict): BUSI sample dictionary, Image to be scaled.
        Returns:
            (dict): scaled image.
        """
        image = sample['image']
        mask = sample['mask']
        coordinates = sample['coordinates']
        cls = sample['cls']
        center = sample['center']

        target_size = [ int(image.size[1]*self.scale), int(image.size[0]*self.scale) ] # (H, W)
        
        return {
            'image' : F.resize(image, target_size, self.interpolation),
            'mask' : F.resize(mask, target_size, self.interpolation),
            'coordinates' : coordinates * self.scale,
            'center' : center * self.scale,
            'cls' : cls
        }

    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        return self.__class__.__name__ + '(scale={0}, interpolation={1})'.format(self.scale, interpolate_str)


class ToTensor(object):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.
    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """
    def __init__(self, normalize=True, target_type='long'):
        self.normalize = normalize
        self.target_type = target_type

    def __call__(self, sample):
        """
        Args:
            sample (dict): BUSI sample dictionary, Image to be scaled.
        Returns:
            (dict): Rescaled image.
        """
        image = sample['image']
        mask = sample['mask']
        coordinates = sample['coordinates'] # [x, y, bbox_w, bbox_h]
        center = sample['center']
        cls = sample['cls']

        tar = np.zeros((mask.size[1], mask.size[0]), dtype=self.target_type)
        tar[np.where(np.array(mask) > 0)] = 1
        
        return {
            'image' : F.to_tensor(image),
            'mask' : torch.from_numpy(tar),
            'coordinates' : torch.from_numpy(coordinates, ),
            'center' : torch.from_numpy(center, ), # [HxW]
            'cls' : torch.tensor(cls, )
        }

    def __repr__(self):
        return self.__class__.__name__ + '()'


class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``input[channel] = (input[channel] - mean[channel]) / std[channel]``
    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        """
        Args:
            sample (dict): BUSI sample dictionary, Image to be scaled.
        Returns:
            (dict): Rescaled image.
        """
        image = sample['image']
        mask = sample['mask']
        
        cls = sample['cls']

        size = np.array([image.size()[2], image.size()[1], image.size()[2], image.size()[1]], dtype=np.float32)
        coordinates = (sample['coordinates'] / size) - 0.5
        
        center = (sample['center'] / np.array([image.size()[1], image.size()[2]], dtype=np.float32)) - 0.5

        return {
            'image' : F.normalize(image, self.mean, self.std),
            'mask' : mask,
            'coordinates' : coordinates,
            'center' : center,
            'cls' : cls
        }

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
