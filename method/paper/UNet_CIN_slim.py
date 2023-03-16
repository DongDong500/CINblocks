# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import unetConv2_cin, unetUp_cin, unetUp_origin, ClassConditionalInstanceNorm2d, unetConv2, unetUp
from .init_weights import init_weights
from torchvision import models


class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, bilinear=True, feature_scale=4, 
                 is_deconv=True, is_batchnorm=True, emb_classes=2, CIN_affine=True):
        super(UNet, self).__init__()        
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.feature_scale = feature_scale
        self.is_deconv = is_deconv
        self.is_batchnorm = is_batchnorm
        self.emb_classes = emb_classes
        self.CIN_affine = CIN_affine
        filters = [64, 128, 256, 512, 1024]

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


    def forward(self, inputs):

        y = inputs['cls']
        input = inputs['image']

        conv1 = self.conv1(input, y)  # 16*512*1024
        maxpool1 = self.maxpool1(conv1)  # 16*256*512

        conv2 = self.conv2(maxpool1, y)  # 32*256*512
        maxpool2 = self.maxpool2(conv2)  # 32*128*256

        conv3 = self.conv3(maxpool2, y)  # 64*128*256
        maxpool3 = self.maxpool3(conv3)  # 64*64*128

        conv4 = self.conv4(maxpool3, y)  # 128*64*128
        maxpool4 = self.maxpool4(conv4)  # 128*32*64

        center = self.center(maxpool4)  # 256*32*64

        up4 = self.up_concat4(center, conv4)  # 128*64*128
        up3 = self.up_concat3(up4, conv3)  # 64*128*256
        up2 = self.up_concat2(up3, conv2)  # 32*256*512
        up1 = self.up_concat1(up2, conv1)  # 16*512*1024

        d1 = self.outconv1(up1)  # 256

        return {
            'image' : d1
        }
