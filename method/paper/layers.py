import torch
import torch.nn as nn
import torch.nn.functional as F
from .init_weights import init_weights


class unetConv2(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, n=2, ks=3, stride=1, padding=1):
        super(unetConv2, self).__init__()
        self.n = n
        self.ks = ks
        self.stride = stride
        self.padding = padding
        s = stride
        p = padding
        
        if is_batchnorm:
            for i in range(1, n + 1):
                conv = nn.Sequential(
                    nn.Conv2d(in_size, out_size, ks, s, p),
                    nn.BatchNorm2d(out_size),
                    nn.ReLU(inplace=True),
                )
                setattr(self, 'conv%d' % i, conv)
                in_size = out_size
        else:
            for i in range(1, n + 1):
                conv = nn.Sequential(
                    nn.Conv2d(in_size, out_size, ks, s, p), 
                    nn.ReLU(inplace=True), 
                )
                setattr(self, 'conv%d' % i, conv)
                in_size = out_size

        # initialise the blocks
        for m in self.children():
            init_weights(m, init_type='kaiming')


    def forward(self, inputs):
        x = inputs
        for i in range(1, self.n + 1):
            conv = getattr(self, 'conv%d' % i)
            x = conv(x)
        return x


class unetUp(nn.Module):
    def __init__(self, in_size, out_size, is_deconv, n_concat=2):
        super(unetUp, self).__init__()
        self.conv = unetConv2(out_size * 2, out_size, False)
        
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=4, stride=2, padding=1)
        else:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)

        # initialise the blocks
        for m in self.children():
            if m.__class__.__name__.find('unetConv2') != -1: continue
            init_weights(m, init_type='kaiming')


    def forward(self, inputs0, *input):
        outputs0 = self.up(inputs0)
        for i in range(len(input)):
            outputs0 = torch.cat([outputs0, input[i]], 1)
        return self.conv(outputs0)

    
class unetUp_origin(nn.Module):
    def __init__(self, in_size, out_size, is_deconv, n_concat=2):
        super(unetUp_origin, self).__init__()
        # self.conv = unetConv2(out_size*2, out_size, False)
        if is_deconv:
            self.conv = unetConv2(in_size + (n_concat - 2) * out_size, out_size, False)
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=4, stride=2, padding=1)
        else:
            self.conv = unetConv2(in_size + (n_concat - 2) * out_size, out_size, False)
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)

        # initialise the blocks
        for m in self.children():
            if m.__class__.__name__.find('unetConv2') != -1: continue
            init_weights(m, init_type='kaiming')

    def forward(self, inputs0, *input):
        # print(self.n_concat)
        # print(input)
        outputs0 = self.up(inputs0)
        for i in range(len(input)):
            outputs0 = torch.cat([outputs0, input[i]], 1)
        return self.conv(outputs0)



### CIN ablation

class ClassConditionalInstanceNorm2d(nn.Module):
    def __init__(self, num_features, num_classes, CIN_affine: bool = True, affine: bool = False):
        """
        Args:
            num_features (int): C from an expected input size (BxCxHxW)
            num_classes (int): 
            affine (bool):  a boolean value that when set to True, 
                `nn.InstanceNorm2d` module has learnable affine parameters, 
                initialized the same way as done for batch normalization. 
                (Default: False)
        """
        super().__init__()
        self.num_features = num_features
        self.isn = nn.InstanceNorm2d(num_features, affine=affine)
        self.embed = nn.Embedding(num_classes, num_features * 2)
        self.CINaffine = CIN_affine
        
        self.mlp = nn.Sequential(
            nn.Linear(num_features * 2, num_features * 2),
            nn.ReLU(inplace=True),
            nn.Linear(num_features * 2, num_features * 2)
        )

        self.embed.weight.data[:, :num_features].normal_(1, 0.02)  # Initialise scale at N(1, 0.02)
        self.embed.weight.data[:, num_features:].zero_()  # Initialise bias at 0

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.1)

    def forward(self, x, y):
        
        out = self.isn(x)
        
        if self.CINaffine:
            gamma, beta = self.mlp(self.embed(y)).chunk(2, 1)
            out = gamma.view(-1, self.num_features, 1, 1) * out + beta.view(-1, self.num_features, 1, 1)
        
        return out

class unetConv2_cin(nn.Module):
    def __init__(self, 
                 in_size, 
                 out_size, 
                 is_batchnorm, 
                 n=2, 
                 ks=3, 
                 stride=1, 
                 padding=1,
                 emb_classes=2,
                 CIN_affine:bool = True):
        
        super(unetConv2_cin, self).__init__()
        self.n = n
        self.ks = ks
        self.stride = stride
        self.padding = padding
        s = stride
        p = padding
        
        if is_batchnorm:
            for i in range(1, n + 1):
                conv = nn.Conv2d(in_size, out_size, ks, s, p)

                cin = ClassConditionalInstanceNorm2d(out_size, emb_classes, CIN_affine)
                act = nn.ReLU(inplace=True)
                
                setattr(self, 'conv%d' % i, conv)
                setattr(self, 'cin%d' % i, cin)
                setattr(self, 'act%d' % i, act)

                in_size = out_size
        else:
            for i in range(1, n + 1):
                conv = nn.Sequential(
                    nn.Conv2d(in_size, out_size, ks, s, p), 
                    nn.ReLU(inplace=True), 
                )
                setattr(self, 'conv%d' % i, conv)
                in_size = out_size

        # initialise the blocks
        for m in self.children():
            init_weights(m, init_type='kaiming')


    def forward(self, inputs, y):
        x = inputs


        for i in range(1, self.n + 1):
            conv = getattr(self, 'conv%d' % i)
            x = conv(x)
            cin = getattr(self, 'cin%d' % i)
            x = cin(x, y)
            act = getattr(self, 'act%d' % i)
            x = act(x)
            
        return x

class unetUp_cin(nn.Module):
    def __init__(self, in_size, out_size, is_deconv, n_concat=2, 
                 emb_classes=2, CIN_affine:bool = True):
        super(unetUp_cin, self).__init__()

        self.conv = unetConv2_cin(out_size * 2, out_size, True, emb_classes, CIN_affine)
        
        if is_deconv is False:
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=4, stride=3, padding=1)
        else:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)

        # initialise the blocks
        for m in self.children():
            if m.__class__.__name__.find('unetConv2') != -1: continue
            init_weights(m, init_type='kaiming')


    def forward(self, inputs0, y, *input):

        outputs0 = self.up(inputs0)

        for i in range(len(input)):
            outputs0 = torch.cat([outputs0, input[i]], 1)
        

        return self.conv(outputs0, y)