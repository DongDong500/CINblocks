from .paper.UNet import UNet as _UNet
from .paper.UNet2Plus import UNet2Plus as _UNet2Plus
from .paper.UNet3Plus import UNet3Plus as _UNet3Plus
from .paper.UNet_CIN import UNet_CIN as _UNet_CIN
from .paper.UNet2Plus_CIN import UNet2Plus_CIN as _UNet2Plus_CIN
from .paper.UNet3Plus_CIN import UNet3Plus_CIN as _UNet3Plus_CIN
from .paper.UNet_CIN_slim import UNet as _UNetCIN_slim


### U-net's base study  ###

def _unet(args, n_channels=3, n_classes=2, verbose=True):
    """For U-net 
    """
    if verbose:
        print("Load Model ")
        print(f"\tU-net, the base model")
    
    return _UNet(
        n_channels=n_channels,
        n_classes=n_classes
    )

def _unet2plus(args, n_channels=3, n_classes=2, verbose=True):
    """For U-net++ 
    """
    if verbose:
        print("Load Model ")
        print(f"\tU-net++, the base model")
    
    return _UNet2Plus(
        n_channels=n_channels,
        n_classes=n_classes,
        is_ds=True
    )

def _unet3plus(args, n_channels=3, n_classes=2, verbose=True):
    """For U-net+++ 
    """
    if verbose:
        print("Load Model ")
        print(f"\tU-net+++, the base model")
    
    return _UNet3Plus(
        n_channels=n_channels,
        n_classes=n_classes,
    )

### U-net's with CIN    ###

def _unet_cin_slim(args, n_channels=3, n_classes=2, verbose=True):
    """For slim U-Net with CIN layers
    """
    if verbose:
        print("Load Model ")
        print(f"\tU-net with CIN (slim version.)")
        print(f"\tEmb classes: {args.emb_classes}, PK affine: {args.CIN_affine}")
    
    return _UNetCIN_slim(
        n_channels=n_channels,
        n_classes=n_classes,
        emb_classes=args.emb_classes,
        CIN_affine=args.CIN_affine
    )

def _unet_cin(args, n_channels=3, n_classes=2, verbose=True):
    """For U-net 
    """
    if verbose:
        print("Load Model ")
        print(f"\tU-net with CIN")
        print(f"\tEmb classes: {args.emb_classes}, PK affine: {args.CIN_affine}")
    
    return _UNet_CIN(
        n_channels=n_channels,
        n_classes=n_classes,
        emb_classes=args.emb_classes,
        CIN_affine=args.CIN_affine
    )

def _unet2plus_cin(args, n_channels=3, n_classes=2, verbose=True):
    """For U-net++ 
    """
    if verbose:
        print("Load Model ")
        print(f"\tU-net 2PLUS with CIN")
        print(f"\tEmb classes: {args.emb_classes}, PK affine: {args.CIN_affine}")
    
    return _UNet2Plus_CIN(
        n_channels=n_channels,
        n_classes=n_classes,
        emb_classes=args.emb_classes,
        CIN_affine=args.CIN_affine,
        is_ds=True
    )

def _unet3plus_cin(args, n_channels=3, n_classes=2, verbose=True):
    """For U-net3Plus
    """
    if verbose:
        print("Load Model ")
        print(f"\tU-net 3PLUS with CIN")
        print(f"\tEmb classes: {args.emb_classes}, PK affine: {args.CIN_affine}")
    
    return _UNet3Plus_CIN(
        n_channels=n_channels,
        n_classes=n_classes,
        emb_classes=args.emb_classes,
        CIN_affine=args.CIN_affine
    )