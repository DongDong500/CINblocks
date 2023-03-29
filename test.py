import torch

import method

from method.paper import cbam


if __name__ == "__main__":

    from torchinfo import summary as sumy_
    from torchsummary import summary

    net = cbam.CBAM(gate_channels=64, no_spatial=True).to('cpu')

    print(summary(model=net, input_size=(64, 256, 256), device='cpu'))

    input_size = (5, 64, 256, 256)
    inputs = torch.rand(input_size)
    print(f'input size: {inputs.size()}')
    print(f'output size: {net(inputs).size()}')