# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import unetConv2, unetUp, unetUp_origin
from .init_weights import init_weights
from torchvision import models


class dUNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, bilinear=True, feature_scale=4, 
                 is_deconv=True, is_batchnorm=True):
        super(dUNet, self).__init__()        
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.feature_scale = feature_scale
        self.is_deconv = is_deconv
        self.is_batchnorm = is_batchnorm
        filters = [64, 128, 256, 512, 1024]

        # downsampling
        self.conv1 = unetConv2(self.n_channels, filters[0], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm)
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
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')


    def dotProduct(self,seg,cls):
        B, N, H, W = seg.size()
        seg = seg.view(B, N, H * W)
        final = torch.einsum("ijk,ij->ijk", [seg, cls])
        final = final.view(B, N, H, W)
        return final


    def forward(self, inputs):

        conv1 = self.conv1(inputs)  # 16*512*1024
        maxpool1 = self.maxpool1(conv1)  # 16*256*512

        conv2 = self.conv2(maxpool1)  # 32*256*512
        maxpool2 = self.maxpool2(conv2)  # 32*128*256

        conv3 = self.conv3(maxpool2)  # 64*128*256
        maxpool3 = self.maxpool3(conv3)  # 64*64*128

        conv4 = self.conv4(maxpool3)  # 128*64*128
        maxpool4 = self.maxpool4(conv4)  # 128*32*64

        center = self.center(maxpool4)  # 256*32*64

        up4 = self.up_concat4(center, conv4)  # 128*64*128
        up3 = self.up_concat3(up4, conv3)  # 64*128*256
        up2 = self.up_concat2(up3, conv2)  # 32*256*512
        up1 = self.up_concat1(up2, conv1)  # 16*512*1024

        d1 = self.outconv1(up1)  # 256

        return d1

class Double_UNet(nn.Module):

    def __init__(self, n_channels=3, n_classes=2, bilinear=True, feature_scale=4, 
                 is_deconv=True, is_batchnorm=True) -> None:
        super().__init__()

        self.unet_first = dUNet(n_channels=n_channels, n_classes=n_classes, bilinear=bilinear, feature_scale=feature_scale, 
                 is_deconv=is_deconv, is_batchnorm=is_batchnorm)
        
        self.unet_second = dUNet(n_channels=n_classes, n_classes=n_classes, bilinear=bilinear, feature_scale=feature_scale, 
                 is_deconv=is_deconv, is_batchnorm=is_batchnorm)

    def forward(self, input):

        x = input['image']

        mid = self.unet_first(x)

        out = self.unet_second(mid)

        return {
            'image' : out
        }