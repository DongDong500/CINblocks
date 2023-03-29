


import torch
import torch.nn as nn


from .unet.unet import UNet


from .cbam import CBAM
from .cin import CINBlock

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

class ConAdapt(nn.Module):

    def __init__(self, n_channels=2, n_classes=2, bilinear=True, feature_scale=4, 
                 is_deconv=True, is_batchnorm=True, emb_classes=2, CIN_affine=True,
                 reduction_ratio=16, 
                 pool_types=['avg', 'max'], 
                 no_spatial=False,
                 use_att=True,
                 use_cin=True):
        
        super().__init__()        
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.feature_scale = feature_scale
        self.is_deconv = is_deconv
        self.is_batchnorm = is_batchnorm
        self.emb_classes = emb_classes
        self.CIN_affine = CIN_affine

        self.use_att = use_att
        self.use_cin = use_cin

        filters = [64, 128, 256, 512, 1024]

        # Attention
        for i in range(len(filters) - 1):
            f = filters[i]
            down_cbam = CBAM(gate_channels=f, reduction_ratio=reduction_ratio, pool_types=pool_types, no_spatial=no_spatial)
            up_cbam = CBAM(gate_channels=f, reduction_ratio=reduction_ratio, pool_types=pool_types, no_spatial=no_spatial)

            setattr(self, 'down_cbam%d' % (i + 1) , down_cbam)
            setattr(self, 'up_cbam%d' % (i + 1) , up_cbam)


        # downsampling
        self.conv1 = unetConv2_cin(self.n_channels, filters[0], self.is_batchnorm, emb_classes=self.emb_classes, CIN_affine=self.CIN_affine)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = unetConv2_cin(filters[0], filters[1], self.is_batchnorm, emb_classes=self.emb_classes, CIN_affine=self.CIN_affine)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = unetConv2_cin(filters[1], filters[2], self.is_batchnorm, emb_classes=self.emb_classes, CIN_affine=self.CIN_affine)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = unetConv2_cin(filters[2], filters[3], self.is_batchnorm, emb_classes=self.emb_classes, CIN_affine=self.CIN_affine)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        self.center = unetConv2(filters[3], filters[4], self.is_batchnorm)

        # upsampling
        self.up_concat4 = unetUp(filters[4], filters[3], self.is_deconv)
        self.up_concat3 = unetUp(filters[3], filters[2], self.is_deconv)
        self.up_concat2 = unetUp(filters[2], filters[1], self.is_deconv)
        self.up_concat1 = unetUp(filters[1], filters[0], self.is_deconv)
        self.outconv1 = nn.Conv2d(filters[0], self.n_classes, 3, padding=1)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, ClassConditionalInstanceNorm2d):
                init_weights(m, init_type='kaiming')


    def dotProduct(self,seg,cls):
        B, N, H, W = seg.size()
        seg = seg.view(B, N, H * W)
        final = torch.einsum("ijk,ij->ijk", [seg, cls])
        final = final.view(B, N, H, W)
        return final


    def forward(self, x, y):

        conv1 = self.conv1(x, y)  # 16*512*1024
        if self.use_att:
            cbam = getattr(self, 'down_cbam1')
            conv1 = cbam(conv1)
        maxpool1 = self.maxpool1(conv1)  # 16*256*512

        conv2 = self.conv2(maxpool1, y)  # 32*256*512
        if self.use_att:
            cbam = getattr(self, 'down_cbam2')
            conv2 = cbam(conv2)
        maxpool2 = self.maxpool2(conv2)  # 32*128*256

        conv3 = self.conv3(maxpool2, y)  # 64*128*256
        if self.use_att:
            cbam = getattr(self, 'down_cbam3')
            conv3 = cbam(conv3)
        maxpool3 = self.maxpool3(conv3)  # 64*64*128

        conv4 = self.conv4(maxpool3, y)  # 128*64*128
        if self.use_att:
            cbam = getattr(self, 'down_cbam4')
            conv4 = cbam(conv4)
        maxpool4 = self.maxpool4(conv4)  # 128*32*64

        center = self.center(maxpool4)  # 256*32*64


        up4 = self.up_concat4(center, conv4)  # 128*64*128
        if self.use_att:
            cbam = getattr(self, 'up_cbam4')
            up4 = cbam(up4)

        up3 = self.up_concat3(up4, conv3)  # 64*128*256
        if self.use_att:
            cbam = getattr(self, 'up_cbam3')
            up3 = cbam(up3)

        up2 = self.up_concat2(up3, conv2)  # 32*256*512
        if self.use_att:
            cbam = getattr(self, 'up_cbam2')
            up2 = cbam(up2)

        up1 = self.up_concat1(up2, conv1)  # 16*512*1024
        if self.use_att:
            cbam = getattr(self, 'up_cbam1')
            up1 = cbam(up1)

        d1 = self.outconv1(up1)  # 256

        return d1





class UNet_ca(nn.Module):

    def __init__(self, n_channels=3, n_classes=2, bilinear=True, feature_scale=4, 
                 is_deconv=True, is_batchnorm=True, 
                 gate_channels=2, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False,
                 num_layer = 6,
                 emb_classes=2,
                 CIN_affine=True,
                 mid_channels=None,
                 use_att=True,
                 use_cin=True) -> None:
        
        super().__init__()

        self.num_layer = num_layer
        self.use_att = use_att
        self.use_cin = use_cin

        self.pt = UNet(n_channels=n_channels, n_classes=n_classes, bilinear=bilinear, feature_scale=feature_scale, 
                 is_deconv=is_deconv, is_batchnorm=is_batchnorm)
        
        self.conadapt = ConAdapt(n_channels=n_classes, n_classes=n_classes, bilinear=bilinear, feature_scale=feature_scale, 
                 is_deconv=is_deconv, is_batchnorm=is_batchnorm,
                 emb_classes=emb_classes, CIN_affine=CIN_affine,
                 reduction_ratio=reduction_ratio, pool_types=pool_types, no_spatial=no_spatial)
        
        # for i in range(1, num_layer + 1):
        #     cbam = CBAM(gate_channels=n_classes, reduction_ratio=reduction_ratio, pool_types=pool_types, no_spatial=no_spatial)
        #     cin = CINBlock(num_features=gate_channels, emb_classes=emb_classes, CIN_affine=CIN_affine, mid_channels=mid_channels)
    
        #     setattr(self, 'cbam%d' % i , cbam)
        #     setattr(self, 'cin%d' % i , cin)

    def forward(self, input):
        
        x = input['image']
        y = input['cls']

        out = self.pt(x)
        
        out = self.conadapt(out, y)

        # for i in range(1, self.num_layer + 1):

        #     if self.use_att:
        #         cbam = getattr(self, 'cbam%d' % i)
        #         out = cbam(out, y)

        #     if self.use_cin:
        #         cin = getattr(self, 'cin%d' % i)
        #         out = cin(out, y)

        return {
            'image' : out
        }
