# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import unetConv2, unetUp_origin
from .init_weights import init_weights
from torchvision import models

from .cin import CINBlock

class UNet2Plus_CIN(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, emb_classes=2, CIN_affine=True, bilinear=True, feature_scale=4, 
                 is_deconv=True, is_batchnorm=True, is_ds=True):
        super(UNet2Plus_CIN, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.feature_scale = feature_scale
        self.is_deconv = is_deconv
        self.is_batchnorm = is_batchnorm
        self.is_ds = is_ds
        filters = [64, 128, 256, 512, 1024]

        # downsampling
        self.conv00 = unetConv2(self.n_channels, filters[0], self.is_batchnorm)
        self.b1 = CINBlock(filters[0], emb_classes, CIN_affine)
        self.maxpool0 = nn.MaxPool2d(kernel_size=2)
        self.conv10 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.b2 = CINBlock(filters[1], emb_classes, CIN_affine)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.conv20 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.b3 = CINBlock(filters[2], emb_classes, CIN_affine)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.conv30 = unetConv2(filters[2], filters[3], self.is_batchnorm)
        self.b4 = CINBlock(filters[3], emb_classes, CIN_affine)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)
        self.conv40 = unetConv2(filters[3], filters[4], self.is_batchnorm)

        # upsampling
        self.up_concat01 = unetUp_origin(filters[1], filters[0], self.is_deconv)
        self.up_concat11 = unetUp_origin(filters[2], filters[1], self.is_deconv)
        self.up_concat21 = unetUp_origin(filters[3], filters[2], self.is_deconv)
        self.up_concat31 = unetUp_origin(filters[4], filters[3], self.is_deconv)

        self.up_concat02 = unetUp_origin(filters[1], filters[0], self.is_deconv, 3)
        self.up_concat12 = unetUp_origin(filters[2], filters[1], self.is_deconv, 3)
        self.up_concat22 = unetUp_origin(filters[3], filters[2], self.is_deconv, 3)

        self.up_concat03 = unetUp_origin(filters[1], filters[0], self.is_deconv, 4)
        self.up_concat13 = unetUp_origin(filters[2], filters[1], self.is_deconv, 4)

        self.up_concat04 = unetUp_origin(filters[1], filters[0], self.is_deconv, 5)

        # upstream CIN block
        self.b5 = CINBlock(filters[3], emb_classes, CIN_affine)
        self.b6 = CINBlock(filters[2], emb_classes, CIN_affine)
        self.b7 = CINBlock(filters[1], emb_classes, CIN_affine)
        self.b8 = CINBlock(filters[0], emb_classes, CIN_affine)

        # final conv (without any concat)
        self.final_1 = nn.Conv2d(filters[0], n_classes, 1)
        self.final_2 = nn.Conv2d(filters[0], n_classes, 1)
        self.final_3 = nn.Conv2d(filters[0], n_classes, 1)
        self.final_4 = nn.Conv2d(filters[0], n_classes, 1)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')


    def forward(self, inputs):
        
        y = inputs['cls']
        inputs = inputs['image']

        # column : 0
        X_00 = self.conv00(inputs)
        X_00 = self.b1(X_00, y)
        maxpool0 = self.maxpool0(X_00)
        X_10 = self.conv10(maxpool0)
        X_10 = self.b2(X_10, y)
        maxpool1 = self.maxpool1(X_10)
        X_20 = self.conv20(maxpool1)
        X_20 = self.b3(X_20, y)
        maxpool2 = self.maxpool2(X_20)
        X_30 = self.conv30(maxpool2)
        X_30 = self.b4(X_30, y)
        maxpool3 = self.maxpool3(X_30)
        X_40 = self.conv40(maxpool3)

        # column : 1
        X_01 = self.up_concat01(X_10, X_00)
        X_11 = self.up_concat11(X_20, X_10)
        X_21 = self.up_concat21(X_30, X_20)
        X_31 = self.up_concat31(X_40, X_30)
        X_31 = self.b5(X_31, y)

        # column : 2
        X_02 = self.up_concat02(X_11, X_00, X_01)
        X_12 = self.up_concat12(X_21, X_10, X_11)
        X_22 = self.up_concat22(X_31, X_20, X_21)
        X_22 = self.b6(X_22, y)

        # column : 3
        X_03 = self.up_concat03(X_12, X_00, X_01, X_02)
        X_13 = self.up_concat13(X_22, X_10, X_11, X_12)
        X_13 = self.b7(X_13, y)

        # column : 4
        X_04 = self.up_concat04(X_13, X_00, X_01, X_02, X_03)
        X_04 = self.b8(X_04, y)

        # final layer
        final_1 = self.final_1(X_01)
        final_2 = self.final_2(X_02)
        final_3 = self.final_3(X_03)
        final_4 = self.final_4(X_04)
        final = (final_1 + final_2 + final_3 + final_4) / 4

        if self.is_ds:
            return {
                'image' : final
            }
        else:
            return {
                'image' : final_4
            }


if __name__ == '__main__':
    model = UNet2Plus_CIN()
    print('# generator parameters:', 1.0 * sum(param.numel() for param in model.parameters()) / 1000000)
    
    params = list(model.named_parameters())
    for i in range(len(params)):
        name, param = params[i]
        print('name:', name, ' param.shape:', param.shape)
