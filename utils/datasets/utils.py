import numpy as np
import torchvision.transforms.functional as F

from matplotlib import pyplot as plt
from PIL import ImageOps


def pad(image, mask, padding_size=(1024, 1024)):
    """Pads the input PIL Image boundaries with zero
    Args:
        image (PIL Image)
        mask (PIL Image)
        padding_size (tuple(int, int)): Desired output size. Size is a sequence like
            (w, h), output size will be matched to this. If image size is larger 
            than the given size, image will be center cropped.
    Returns:
        image (PIL Image)
        mask (PIL Image)
    """
    # Center crop
    if image.size[0] > padding_size[0] or image.size[1] > padding_size[1]:
        w = image.size[0]
        h = image.size[1]
        # width
        if w > padding_size[0]:
            image = image.crop( (int(w/2) - padding_size[0]/2, 0, int(w/2) + padding_size[0]/2, h) )
            mask = mask.crop( (int(w/2) - padding_size[0]/2, 0, int(w/2) + padding_size[0]/2, h) )
        # height
        if h > padding_size[1]:
            image = image.crop( (0, int(h/2) - padding_size[1]/2, w, int(h/2) + padding_size[1]/2) )
            mask = mask.crop( (0, int(h/2) - padding_size[1]/2, w, int(h/2) + padding_size[1]/2) )
        
    # Padding
    width = padding_size[0] - image.size[0]
    height = padding_size[1] - image.size[1]

    if width % 2 == 0 and width > 0:
        left, right = width/2, width/2
    elif width % 2 == 1:
        left, right = (width + 1)/2, (width - 1)/2
    else:
        left, right = 0, 0

    if height % 2 == 0 and height > 0:
        top, bottom = height/2, height/2
    elif height % 2 == 1:
        top, bottom = (height + 1)/2, (height - 1)/2
    else:
        top, bottom = 0, 0
    
    border = (int(left), int(top), int(right), int(bottom))

    image = ImageOps.expand(image, border, fill=0)
    mask = ImageOps.expand(mask, border, fill=0)

    if image.size != padding_size:
        raise Exception("Padding size does not match", image.size, padding_size)

    return image, mask

def show_sample(busi: dict, sample: dict, verbose: bool=False):
    """Show sample image
    Args:
        busi (dict): {
            'image' : (PIL.Image),
            'mask' : (PIL.Image),
            'coordinates' : (np.array),
            'height' : (int),
            'width' : (int),
            'category' : (str),
            'id' : (str, int) index of the corresponding image
        }
        sample (dict): {
            'image' : (Tensor) image,
            'mask' : (Tensor) mask,
            'coordinates' : (Tensor) [x, y, bbox_w, bbox_h] normalized coordinates.
            'cls' : ,
            'center' : 
        }
    """
    denorm = Denormalize()
    cmap = color_map()

    image = sample['image']
    mask = sample['mask']
    coordinates = sample['coordinates']
    center = sample['center']
    
    size = image.size()
    r = np.array([size[2], size[1], size[2], size[1]], dtype=np.float32)
    x, y, w, h = ((coordinates.numpy() + 0.5) * r).astype(np.int32)

    r = np.array([size[2], size[1]], dtype=np.float32)
    x_c, y_c = ((center.numpy() + 0.5) * r).astype(np.int32)

    img = (denorm(image.numpy()) * 255).transpose(1, 2, 0).astype(np.uint8)
        
    tar = np.zeros(shape=(mask.shape), dtype=np.uint8)
    tar[mask.numpy() > 0] = 1
    tar[y:y+h, x:x+2] = 2
    tar[y:y+h, x+w-2:x+w] = 2
    tar[y:y+2, x:x+w] = 2
    tar[y+h-2:y+h, x:x+w] = 2

    tar[x_c-2:x_c+2, y_c-20: y_c+20] = 2
    tar[x_c-20:x_c+20, y_c-2: y_c+2] = 2

    if verbose:
        print('coordinate', x, y, w, h)
        print('tar shape', tar.shape)
    
    x, y, w, h = busi['coordinates'].astype(np.int32)
    lbl = np.zeros(shape=(busi['height'], busi['width']), dtype=np.uint8)
    lbl[np.array(busi['mask'], dtype=np.uint8) > 0] = 1
    lbl[y:y+h, x:x+2] = 2
    lbl[y:y+h, x+w-2:x+w] = 2
    lbl[y:y+2, x:x+w] = 2
    lbl[y+h-2:y+h, x:x+w] = 2

    if verbose:
        print('coordinate', x, y, w, h)
        print('lbl shape', lbl.shape)

    fig = plt.figure(figsize=(6, 9))

    ax1 = fig.add_subplot(3, 2, 1)
    ax1.imshow(img, )
    ax1.set_xlabel('Image')

    ax2 = fig.add_subplot(3, 2, 2)
    ax2.imshow(cmap[mask.numpy()], )
    ax2.set_xlabel('Mask')

    ax3 = fig.add_subplot(3, 2, 3)
    ax3.imshow(cmap[tar], )
    ax3.set_xlabel('bbox')

    ax4 = fig.add_subplot(3, 2, 4)
    ax4.imshow(cmap[lbl], )
    ax4.set_xlabel('Original bbox')

    ax5 = fig.add_subplot(3, 2, 5)
    ax5.imshow(busi['image'], )
    ax5.set_xlabel(f"Original image {busi['category']} ({busi['id']})")

    ax6 = fig.add_subplot(3, 2, 6)
    ax6.imshow(busi['mask'], cmap='gray')
    ax6.set_xlabel('Original mask')

    plt.show()

def show_image(src: dict, tar: dict, verbose: bool=False):
    """Show result image
    Args:
        src (dict): {
            'image' : (Tensor) image,
            'center' : 
        }
        tar (dict): {
            'image' : ,
            'mask' : ,
            'coordinates' : ,
            'center' : ,
            'cls' : 
        }
    """
    denorm = Denormalize()
    cmap = color_map()

    pm = src['image']
    pc = src['center']

    tm = tar['mask']
    tc = tar['center']

    r = np.array([224, 224], dtype=np.float32)
    px, py = ((pc + 0.5) * r).astype(np.int32)
   
    pred = np.zeros(shape=(pm.shape), dtype=np.uint8)
    pred[pm > 0] = 1

    pred[py-10 : py+10, px-2 : px+2] = 2
    pred[py-2 : py+2, px-10 : px+10] = 2

    tx, ty = ((tc + 0.5) * r).astype(np.int32)

    true = np.zeros(shape=(tm.shape), dtype=np.uint8)
    true[tm > 0] = 1

    true[ty-10 : ty+10, tx-2 : tx+2] = 2
    true[ty-2 : ty+2, tx-10 : tx+10] = 2

    fig = plt.figure(figsize=(10, 5)) # (W, H)

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(cmap[true], )
    ax1.set_xlabel('True mask')

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.imshow(cmap[pred], )
    ax2.set_xlabel('Pred mask')

    plt.show()

def color_map(N=256, normalized=False, preds=False):
    """Color map 2d-uint8 np.array will be matched to (r, g, b)
    """
    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)

    for i in range(N):
        r = g = b = 0
        if preds:
            cmap[i] = np.array([255, i, i])
        else:
            cmap[i] = np.array([i, i, 255])
    
    cmap[0] = np.array([255, 255, 255])
    cmap[1] = np.array([255, 102, 102]) if preds else np.array([102, 178, 255])
    cmap[2] = np.array([0, 0, 255]) if preds else np.array([255, 0, 0])
    cmap = cmap/255 if normalized else cmap
    
    return cmap

class Denormalize(object):

    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        mean = np.array(mean, dtype=np.float32)
        std = np.array(std, dtype=np.float32)
        self._mean = -mean/std
        self._std = 1/std

    def __call__(self, tensor):
        if isinstance(tensor, np.ndarray):
            return (tensor - self._mean.reshape(-1,1,1)) / self._std.reshape(-1,1,1)
        return F.normalize(tensor, self._mean, self._std)