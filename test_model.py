import torch

from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights

import method
from method.utils.custom_summary import summary

from torchinfo import summary as sumy_


if __name__ == "__main__":

    # Model (nn.Module)
    # model = maskrcnn_resnet50_fpn(
    #     weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT
    # )
    # model = method.method.__dict__["Unet_"](
    #     n_channels=3,
    #     n_classes=2
    # )
    from main import get_argparser


    test_model = [
        '_unet_cin_slim',
    ]

    for tm in test_model:
            
        args = get_argparser()
        model = method.method.__dict__[tm](
            args
        )

        print('# generator parameters:', 1.0 * sum(param.numel() for param in model.parameters()) / 1000000)
        
        params = list(model.named_parameters())
        for i in range(len(params)):
            name, param = params[i]
            print('name:', name, ' param.shape:', param.shape)
        
        print("Done")

        inputs = {
            'image' : torch.rand(5, 3, 256, 256),
            'center' : torch.rand(5, 2),
            'cls' : torch.randint(low=0, high=2, size=(5, ))
        }

        try:
            out = model(inputs['image'])
        except:
            out = model(inputs)

        print("Output type:", type(out))

        if isinstance(out, dict):
            
            for k, v in out.items():
                s = v.size()

                print(f"\t{k}:", list(s))
        else:
            print(f"\t{out.size()}")
        
        try:
            print(summary(
                model=model,
                input_size=inputs
            ))
        except:
            print(sumy_(
                model=model,
                input_size=(5, 3, 256, 256),
            ))
