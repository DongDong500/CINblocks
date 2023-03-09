# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import unetConv2, unetUp, unetUp_origin
from .init_weights import init_weights
from torchvision import models

from .cin import CINBlock

class UNet_CIN(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, emb_classes=2, CIN_affine=True, bilinear=True, feature_scale=4, 
                 is_deconv=True, is_batchnorm=True):
        super(UNet_CIN, self).__init__()        
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.feature_scale = feature_scale
        self.is_deconv = is_deconv
        self.is_batchnorm = is_batchnorm
        filters = [64, 128, 256, 512, 1024]

        # downsampling
        self.conv1 = unetConv2(self.n_channels, filters[0], self.is_batchnorm)
        self.b1 = CINBlock(filters[0], emb_classes, CIN_affine)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.b2 = CINBlock(filters[1], emb_classes, CIN_affine)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.b3 = CINBlock(filters[2], emb_classes, CIN_affine)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm)
        self.b4 = CINBlock(filters[3], emb_classes, CIN_affine)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        self.center = unetConv2(filters[3], filters[4], self.is_batchnorm)

        # upsampling
        self.up_concat4 = unetUp(filters[4], filters[3], self.is_deconv)
        self.b5 = CINBlock(filters[3], emb_classes, CIN_affine)
        self.up_concat3 = unetUp(filters[3], filters[2], self.is_deconv)
        self.b6 = CINBlock(filters[2], emb_classes, CIN_affine)
        self.up_concat2 = unetUp(filters[2], filters[1], self.is_deconv)
        self.b7 = CINBlock(filters[1], emb_classes, CIN_affine)
        self.up_concat1 = unetUp(filters[1], filters[0], self.is_deconv)
        self.b8 = CINBlock(filters[0], emb_classes, CIN_affine)

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
        
        y = inputs['cls']
        inputs = inputs['image']

        conv1 = self.conv1(inputs)  # 16*512*1024
        conv1 = self.b1(conv1, y)
        maxpool1 = self.maxpool1(conv1)  # 16*256*512

        conv2 = self.conv2(maxpool1)  # 32*256*512
        conv2 = self.b2(conv2, y)
        maxpool2 = self.maxpool2(conv2)  # 32*128*256

        conv3 = self.conv3(maxpool2)  # 64*128*256
        conv3 = self.b3(conv3, y)
        maxpool3 = self.maxpool3(conv3)  # 64*64*128

        conv4 = self.conv4(maxpool3)  # 128*64*128
        conv4 = self.b4(conv4, y)
        maxpool4 = self.maxpool4(conv4)  # 128*32*64

        center = self.center(maxpool4)  # 256*32*64

        up4 = self.up_concat4(center, conv4)  # 128*64*128
        up4 = self.b5(up4, y)
        up3 = self.up_concat3(up4, conv3)  # 64*128*256
        up3 = self.b6(up3, y)
        up2 = self.up_concat2(up3, conv2)  # 32*256*512
        up2 = self.b7(up2, y)
        up1 = self.up_concat1(up2, conv1)  # 16*512*1024
        up1 = self.b8(up1, y)
        d1 = self.outconv1(up1)  # 256


        return {
            'image' : d1
        }
