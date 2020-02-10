# -*- coding: utf-8 -*-

from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchvision.models.vgg import VGG

class ResidualDenseBLock(nn.Module):

    def __init__(self, in_channels, out_channels, opt):

        super(ResidualDenseBLock, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.conv1_1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.conv1_3 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.conv1_4 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.cat_1 = nn.Conv3d(out_channels*3,out_channels, kernel_size=3, padding=1)
        if opt == "encoder":
            self.final = nn.MaxPool3d(kernel_size=2, stride=2)
        else:
            self.final = nn.ConvTranspose3d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)


    def forward(self, x):

        ya = self.bn1(self.relu(self.conv1_1(x)))
        yb = self.bn1(self.relu(self.conv1_2(ya)))
        yc = self.bn1(self.relu(self.conv1_2(ya+yb)))
        yd = self.bn1(self.relu(self.conv1_2(ya+yb+yc)))

        concat = torch.cat([yb, yc, yd], 1)
        y = ya + self.bn1(self.relu(self.cat_1(concat)))
        y = self.final(y)

        return y

class MiddleCodification(nn.Module):

    def __init__(self, channels):
        super(MiddleCodification, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv_1 = nn.Conv3d(channels, 2*channels, kernel_size=3, padding=1)
        self.conv_2 = nn.Conv3d(2*channels, 2*channels, kernel_size=3, padding=1)
        self.conv_3 = nn.Conv3d(2*channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(channels)
        self.bn2 = nn.BatchNorm3d(2*channels)

    def forward(self, x):

        y = self.bn2(self.relu(self.conv_1(x)))
        y = self.bn2(self.relu(self.conv_2(y)))
        y = self.bn1(self.relu(self.conv_3(y)))

        return y


class L4_DENSE_RED_3D(nn.Module):

    def __init__(self):

        super().__init__()
        self.res1 = ResidualDenseBLock(1,64, "encoder")
        self.res2 = ResidualDenseBLock(64, 128, "encoder")
        self.res3 = ResidualDenseBLock(128, 256, "encoder")
        self.res4 = ResidualDenseBLock(256, 512, "encoder")
        self.middle = MiddleCodification(512)
        self.res5 = ResidualDenseBLock(512, 256, "decoder")
        self.res6 = ResidualDenseBLock(256, 128, "decoder")
        self.res7 = ResidualDenseBLock(128,64, "decoder")
        self.res8 = ResidualDenseBLock(64, 1, "decoder")

    def forward(self, x):

        y1 = self.res1(x)
        y2 = self.res2(y1)
        y3 = self.res3(y2)
        y4 = self.res4(y3)
        y5 = self.middle(y4)
        y7 = self.res5(y5+y4)
        y8 = self.res6(y7+y3)
        y9 = self.res7(y8+y2)
        y10 = self.res8(y9+y1)

        return y10


class L3_DENSE_RED_3D(nn.Module):

    def __init__(self):

        super().__init__()
        self.res1 = ResidualDenseBLock(1,64, "encoder")
        self.res2 = ResidualDenseBLock(64, 128, "encoder")
        self.res3 = ResidualDenseBLock(128, 256, "encoder")
        self.middle = MiddleCodification(256)
        self.res4 = ResidualDenseBLock(256, 128, "decoder")
        self.res5 = ResidualDenseBLock(128,64, "decoder")
        self.res6 = ResidualDenseBLock(64, 1, "decoder")

    def forward(self, x):

        y1 = self.res1(x)
        y2 = self.res2(y1)
        y3 = self.res3(y2)
        y4 = self.middle(y3)
        y5 = self.res4(y4+y3)
        y6 = self.res5(y5+y2)
        y7 = self.res6(y6+y1)

        return y7

class L2_DENSE_RED_3D(nn.Module):

    def __init__(self):

        super().__init__()
        self.res1 = ResidualDenseBLock(1,64, "encoder")
        self.res2 = ResidualDenseBLock(64, 128, "encoder")
        self.middle = MiddleCodification(128)
        self.res3 = ResidualDenseBLock(128,64, "decoder")
        self.res4 = ResidualDenseBLock(64, 1, "decoder")

    def forward(self, x):

        y1 = self.res1(x)
        y2 = self.res2(y1)
        y3 = self.middle(y2)
        y4 = self.res3(y3+y2)
        y5 = self.res4(y4+y1)

        return y5

class L1_DENSE_RED_3D(nn.Module):

    def __init__(self):

        super().__init__()
        self.res1 = ResidualDenseBLock(1,64, "encoder")
        self.middle = MiddleCodification(64)
        self.res2 = ResidualDenseBLock(64,1, "decoder")

    def forward(self, x):

        y1 = self.res1(x)
        y2 = self.middle(y1)
        y3 = self.res2(y1+y2)

        return y3

class FULLY_DENSE_UNET_3D(nn.Module):

    def __init__(self):

        super().__init__()
        #encoder
        self.relu = nn.ReLU(inplace=True)
        self.conv1_1 = nn.Conv3d(1, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv3d(64, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(64)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        self.conv2_1 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv3d(192, 128, kernel_size=3, padding=1)
        self.conv2_3 = nn.Conv3d(320, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(128)

        self.conv3_1 = nn.Conv3d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv3d(384, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv3d(640, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm3d(256)

        self.conv4_1 = nn.Conv3d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv3d(768, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv3d(1280, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm3d(512)

        #middle
        self.conv5_1 = nn.Conv3d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv3d(512, 1024, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv3d(1024, 512, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm3d(1024)

        #decoder
        self.deconv6_1 = nn.ConvTranspose3d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.conv6_2 = nn.Conv3d(1024, 512, kernel_size=3, padding=1)
        self.conv6_3 = nn.Conv3d(1024, 512, kernel_size=3, padding=1)
        self.conv6_4 = nn.Conv3d(1536, 256, kernel_size=3, padding=1)

        self.deconv7_1 = nn.ConvTranspose3d(256, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.conv7_2 = nn.Conv3d(512, 256, kernel_size=3, padding=1)
        self.conv7_3 = nn.Conv3d(512, 256, kernel_size=3, padding=1)
        self.conv7_4 = nn.Conv3d(768, 128, kernel_size=3, padding=1)

        self.deconv8_1 = nn.ConvTranspose3d(128, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.conv8_2 = nn.Conv3d(256, 128, kernel_size=3, padding=1)
        self.conv8_3 = nn.Conv3d(256, 128, kernel_size=3, padding=1)
        self.conv8_4 = nn.Conv3d(384, 64, kernel_size=3, padding=1)

        self.deconv9_1 = nn.ConvTranspose3d(64, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.conv9_2 = nn.Conv3d(128, 64, kernel_size=3, padding=1)
        self.conv9_3 = nn.Conv3d(64, 64, kernel_size=3, padding=1)
        self.conv9_4 = nn.Conv3d(64, 1, kernel_size=1)
        self.outputer = nn.Tanh()

    def forward(self, x):

        #encoder
        #4x1x16x128x128
        y = self.bn1(self.relu(self.conv1_1(x))) #4x64x16x128x128
        y = self.bn1(self.relu(self.conv1_2(y))) #4x64x16x128x128
        y1 = y #4x64x16x128x128
        y = self.pool(y) #4x64x8x64x64


        ya = self.bn2(self.relu(self.conv2_1(y)))   #4x128x8x64x64
        concat = torch.cat([y,ya], 1)
        yb = self.bn2(self.relu(self.conv2_2(concat))) #4x128x8x64x64
        concat = torch.cat([y, ya, yb], 1)
        yc = self.bn2(self.relu(self.conv2_3(concat))) #4x128x8x64x64
        y2 = yc #4x128x8x64x64

        yg = self.pool(y2) #4x128x4x32x32
        yh = self.bn3(self.relu(self.conv3_1(yg))) #4x256x4x32x32
        concat = torch.cat([yg, yh], 1)
        yi = self.bn3(self.relu(self.conv3_2(concat))) #4x256x4x32x32
        concat = torch.cat([yi, yg, yh], 1)
        yj = self.bn3(self.relu(self.conv3_3(concat))) #4x256x4x32x32
        y3 = yj #4x256x4x32x32

        yk = self.pool(y3) #4x256x2x16x16
        yl = self.bn4(self.relu(self.conv4_1(yk))) #4x512x2x16x16
        concat = torch.cat([yk, yl], 1)
        ym = self.bn4(self.relu(self.conv4_2(concat))) #4x512x2x16x16
        concat = torch.cat([ym, yk, yl], 1)
        yn = self.bn4(self.relu(self.conv4_3(concat))) #4x512x2x16x16
        y4 = yn #4x512x2x16x16

        #middle
        y = self.pool(y4) #4x512x1x8x8
        y = self.bn4(self.relu(self.conv5_1(y))) #4x512x1x8x8
        y = self.bn5(self.relu(self.conv5_2(y))) #4x1024x1x8x8
        y = self.bn4(self.relu(self.conv5_3(y))) #4x512x1x8x8

        #decoder
        y = self.bn4(self.relu(self.deconv6_1(y))) #4x512x2x16x16
        concat = torch.cat([y,y4], 1) #y + y4 #4x512x2x16x16
        yo = self.bn4(self.relu(self.conv6_2(concat)))  #4x512x2x16x16
        concat = torch.cat([y, yo], 1)
        yp = self.bn4(self.relu(self.conv6_3(concat))) #4x512x2x16x16
        concat = torch.cat([y, yo, yp], 1)
        yq = self.bn3(self.relu(self.conv6_4(concat))) #4x256x2x16x16


        y = self.bn3(self.relu(self.deconv7_1(yq))) #4x256x4x32x32

        concat = torch.cat([y,y3], 1) #y = y + y3 #4x256x4x32x32
        yr = self.bn3(self.relu(self.conv7_2(concat))) #4x256x4x32x32
        concat = torch.cat([y, yr], 1)
        ys = self.bn3(self.relu(self.conv7_3(concat))) #4x256x4x32x32
        concat = torch.cat([y, yr, ys], 1)
        yt = self.bn2(self.relu(self.conv7_4(concat))) #4x128x4x32x32


        y = self.bn2(self.relu(self.deconv8_1(yt))) #4x128x8x64x64
        concat = torch.cat([y, y2], 1) #y = y + y2 #4x128x8x64x64
        yu = self.bn2(self.relu(self.conv8_2(concat))) #4x128x8x64x64
        concat = torch.cat([y, yu], 1)
        yv = self.bn2(self.relu(self.conv8_3(concat))) #4x128x8x64x64
        concat = torch.cat([y, yu, yv], 1)
        yx = self.bn1(self.relu(self.conv8_4(concat))) #4x64x8x64x64


        y = self.bn1(self.relu(self.deconv9_1(yx))) #4x64x16x128x128
        concat = torch.cat([y, y1], 1)#y = y + y1
        yy = self.bn1(self.relu(self.conv9_2(concat)))
        yw = self.bn1(self.relu(self.conv9_3(yy)))
        y = self.outputer(self.relu(self.conv9_4(yw)))

        return y

class DENSE_UNET_3D(nn.Module):

    def __init__(self):

        super().__init__()
        #encoder
        self.relu = nn.ReLU(inplace=True)
        self.conv1_1 = nn.Conv3d(1, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv3d(64, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(64)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        self.conv2_1 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv3d(192, 128, kernel_size=3, padding=1)
        self.conv2_3 = nn.Conv3d(320, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(128)

        self.conv3_1 = nn.Conv3d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv3d(384, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv3d(640, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm3d(256)

        self.conv4_1 = nn.Conv3d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv3d(768, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv3d(1280, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm3d(512)

        #middle
        self.conv5_1 = nn.Conv3d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv3d(512, 1024, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv3d(1024, 512, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm3d(1024)

        #decoder
        self.deconv6_1 = nn.ConvTranspose3d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.conv6_2 = nn.Conv3d(512, 512, kernel_size=3, padding=1)
        self.conv6_3 = nn.Conv3d(1024, 512, kernel_size=3, padding=1)
        self.conv6_4 = nn.Conv3d(1536, 256, kernel_size=3, padding=1)

        self.deconv7_1 = nn.ConvTranspose3d(256, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.conv7_2 = nn.Conv3d(256, 256, kernel_size=3, padding=1)
        self.conv7_3 = nn.Conv3d(512, 256, kernel_size=3, padding=1)
        self.conv7_4 = nn.Conv3d(768, 128, kernel_size=3, padding=1)

        self.deconv8_1 = nn.ConvTranspose3d(128, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.conv8_2 = nn.Conv3d(128, 128, kernel_size=3, padding=1)
        self.conv8_3 = nn.Conv3d(256, 128, kernel_size=3, padding=1)
        self.conv8_4 = nn.Conv3d(384, 64, kernel_size=3, padding=1)

        self.deconv9_1 = nn.ConvTranspose3d(64, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.conv9_2 = nn.Conv3d(64, 64, kernel_size=3, padding=1)
        self.conv9_3 = nn.Conv3d(64, 64, kernel_size=3, padding=1)
        self.conv9_4 = nn.Conv3d(64, 1, kernel_size=1)
        self.outputer = nn.Tanh()

    def forward(self, x):

        #encoder
        #4x1x16x128x128
        y = self.bn1(self.relu(self.conv1_1(x))) #4x64x16x128x128
        y = self.bn1(self.relu(self.conv1_2(y))) #4x64x16x128x128
        y1 = y #4x64x16x128x128
        y = self.pool(y) #4x64x8x64x64


        ya = self.bn2(self.relu(self.conv2_1(y)))   #4x128x8x64x64
        concat = torch.cat([y,ya], 1)
        yb = self.bn2(self.relu(self.conv2_2(concat))) #4x128x8x64x64
        concat = torch.cat([y, ya, yb], 1)
        yc = self.bn2(self.relu(self.conv2_3(concat))) #4x128x8x64x64
        y2 = yc #4x128x8x64x64

        yg = self.pool(y2) #4x128x4x32x32
        yh = self.bn3(self.relu(self.conv3_1(yg))) #4x256x4x32x32
        concat = torch.cat([yg, yh], 1)
        yi = self.bn3(self.relu(self.conv3_2(concat))) #4x256x4x32x32
        concat = torch.cat([yi, yg, yh], 1)
        yj = self.bn3(self.relu(self.conv3_3(concat))) #4x256x4x32x32
        y3 = yj #4x256x4x32x32

        yk = self.pool(y3) #4x256x2x16x16
        yl = self.bn4(self.relu(self.conv4_1(yk))) #4x512x2x16x16
        concat = torch.cat([yk, yl], 1)
        ym = self.bn4(self.relu(self.conv4_2(concat))) #4x512x2x16x16
        concat = torch.cat([ym, yk, yl], 1)
        yn = self.bn4(self.relu(self.conv4_3(concat))) #4x512x2x16x16
        y4 = yn #4x512x2x16x16

        #middle
        y = self.pool(y4) #4x512x1x8x8
        y = self.bn4(self.relu(self.conv5_1(y))) #4x512x1x8x8
        y = self.bn5(self.relu(self.conv5_2(y))) #4x1024x1x8x8
        y = self.bn4(self.relu(self.conv5_3(y))) #4x512x1x8x8

        #decoder
        y = self.bn4(self.relu(self.deconv6_1(y))) #4x512x2x16x16
        y = y + y4 #4x512x2x16x16
        yo = self.bn4(self.relu(self.conv6_2(y)))  #4x512x2x16x16
        concat = torch.cat([y, yo], 1)
        yp = self.bn4(self.relu(self.conv6_3(concat))) #4x512x2x16x16
        concat = torch.cat([y, yo, yp], 1)
        yq = self.bn3(self.relu(self.conv6_4(concat))) #4x256x2x16x16


        y = self.bn3(self.relu(self.deconv7_1(yq))) #4x256x4x32x32
        y = y + y3 #4x256x4x32x32
        yr = self.bn3(self.relu(self.conv7_2(y))) #4x256x4x32x32
        concat = torch.cat([y, yr], 1)
        ys = self.bn3(self.relu(self.conv7_3(concat))) #4x256x4x32x32
        concat = torch.cat([y, yr, ys], 1)
        yt = self.bn2(self.relu(self.conv7_4(concat))) #4x128x4x32x32


        y = self.bn2(self.relu(self.deconv8_1(yt))) #4x128x8x64x64
        y = y + y2 #4x128x8x64x64
        yu = self.bn2(self.relu(self.conv8_2(y))) #4x128x8x64x64
        concat = torch.cat([y, yu], 1)
        yv = self.bn2(self.relu(self.conv8_3(concat))) #4x128x8x64x64
        concat = torch.cat([y, yu, yv], 1)
        yx = self.bn1(self.relu(self.conv8_4(concat))) #4x64x8x64x64


        y = self.bn1(self.relu(self.deconv9_1(yx))) #4x64x16x128x128
        y = y + y1
        yy = self.bn1(self.relu(self.conv9_2(y)))
        yw = self.bn1(self.relu(self.conv9_3(yy)))
        y = self.outputer(self.relu(self.conv9_4(yw)))

        return y

class UNET_3D(nn.Module):

    def __init__(self):

        super().__init__()
        #encoder
        self.relu = nn.ReLU(inplace=True)
        self.conv1_1 = nn.Conv3d(1, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv3d(64, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(64)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        self.conv2_1 = nn.Conv3d(64, 64, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.conv2_3 = nn.Conv3d(128, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(128)

        self.conv3_1 = nn.Conv3d(128, 128, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv3d(128, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv3d(256, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm3d(256)

        self.conv4_1 = nn.Conv3d(256, 256, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv3d(256, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv3d(512, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm3d(512)

        #middle
        self.conv5_1 = nn.Conv3d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv3d(512, 1024, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv3d(1024, 512, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm3d(1024)

        #decoder
        self.deconv6_1 = nn.ConvTranspose3d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.conv6_2 = nn.Conv3d(512, 512, kernel_size=3, padding=1)
        self.conv6_3 = nn.Conv3d(512, 512, kernel_size=3, padding=1)
        self.conv6_4 = nn.Conv3d(512, 256, kernel_size=3, padding=1)

        self.deconv7_1 = nn.ConvTranspose3d(256, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.conv7_2 = nn.Conv3d(256, 256, kernel_size=3, padding=1)
        self.conv7_3 = nn.Conv3d(256, 256, kernel_size=3, padding=1)
        self.conv7_4 = nn.Conv3d(256, 128, kernel_size=3, padding=1)

        self.deconv8_1 = nn.ConvTranspose3d(128, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.conv8_2 = nn.Conv3d(128, 128, kernel_size=3, padding=1)
        self.conv8_3 = nn.Conv3d(128, 128, kernel_size=3, padding=1)
        self.conv8_4 = nn.Conv3d(128, 64, kernel_size=3, padding=1)

        self.deconv9_1 = nn.ConvTranspose3d(64, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.conv9_2 = nn.Conv3d(64, 64, kernel_size=3, padding=1)
        self.conv9_3 = nn.Conv3d(64, 64, kernel_size=3, padding=1)
        self.conv9_4 = nn.Conv3d(64, 1, kernel_size=1)
        self.outputer = nn.Tanh()

    def forward(self, x):

        #encoder
        #in 256x256
        y = self.bn1(self.relu(self.conv1_1(x)))
        y1 = self.bn1(self.relu(self.conv1_2(y)))
        y = self.pool(y1)
        #in 128x128
        y = self.bn1(self.relu(self.conv2_1(y)))
        y = self.bn2(self.relu(self.conv2_2(y)))
        y2 = self.bn2(self.relu(self.conv2_3(y)))
        y = self.pool(y2)
        #in 64x64
        y = self.bn2(self.relu(self.conv3_1(y)))
        y = self.bn3(self.relu(self.conv3_2(y)))
        y3 = self.bn3(self.relu(self.conv3_3(y)))
        y = self.pool(y3)
        #in 32x32
        y = self.bn3(self.relu(self.conv4_1(y)))
        y = self.bn4(self.relu(self.conv4_2(y)))
        y4 = self.bn4(self.relu(self.conv4_3(y)))
        y = self.pool(y4)
        #in 16x16
        #middle
        y = self.bn4(self.relu(self.conv5_1(y)))
        y = self.bn5(self.relu(self.conv5_2(y)))
        y = self.bn4(self.relu(self.conv5_3(y)))

        #decoder


        y = self.bn4(self.relu(self.deconv6_1(y)))
        y = y + y4
        y = self.bn4(self.relu(self.conv6_2(y)))
        y = self.bn4(self.relu(self.conv6_3(y)))
        y = self.bn3(self.relu(self.conv6_4(y)))


        y = self.bn3(self.relu(self.deconv7_1(y)))
        y = y + y3
        y = self.bn3(self.relu(self.conv7_2(y)))
        y = self.bn3(self.relu(self.conv7_3(y)))
        y = self.bn2(self.relu(self.conv7_4(y)))


        y = self.bn2(self.relu(self.deconv8_1(y)))
        y = y + y2
        y = self.bn2(self.relu(self.conv8_2(y)))
        y = self.bn2(self.relu(self.conv8_3(y)))
        y = self.bn1(self.relu(self.conv8_4(y)))


        y = self.bn1(self.relu(self.deconv9_1(y)))
        y = y + y1
        y = self.bn1(self.relu(self.conv9_2(y)))
        y = self.bn1(self.relu(self.conv9_3(y)))
        y = self.outputer(self.relu(self.conv9_4(y)))

        return y

class UNETmodel(nn.Module):

    def __init__(self, pretrained_net):
        super().__init__()

        self.pretrained_net = pretrained_net
        self.relu    = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1     = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2     = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3     = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4     = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5     = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, 1, kernel_size=1)
        self.outputer = nn.Tanh()

    def forward(self, x):
        output = self.pretrained_net(x)
        x5 = output['x5']  # size=(N, 512, x.H/32, x.W/32)
        x4 = output['x4']  # size=(N, 512, x.H/16, x.W/16)
        x3 = output['x3']  # size=(N, 256, x.H/8,  x.W/8)
        x2 = output['x2']  # size=(N, 128, x.H/4,  x.W/4)
        x1 = output['x1']  # size=(N, 64, x.H/2,  x.W/2)

        score = self.bn1(self.relu(self.deconv1(x5)))     # size=(N, 512, x.H/16, x.W/16)
        score = score + x4                                # element-wise add, size=(N, 512, x.H/16, x.W/16)
        score = self.bn2(self.relu(self.deconv2(score)))  # size=(N, 256, x.H/8, x.W/8)
        score = score + x3                                # element-wise add, size=(N, 256, x.H/8, x.W/8)
        score = self.bn3(self.relu(self.deconv3(score)))  # size=(N, 128, x.H/4, x.W/4)
        score = score + x2                                # element-wise add, size=(N, 128, x.H/4, x.W/4)
        score = self.bn4(self.relu(self.deconv4(score)))  # size=(N, 64, x.H/2, x.W/2)
        score = score + x1                                # element-wise add, size=(N, 64, x.H/2, x.W/2)
        score = self.bn5(self.relu(self.deconv5(score)))  # size=(N, 32, x.H, x.W)
        score = self.classifier(score)                    # size=(N, n_class, x.H/1, x.W/1)
        score = self.outputer(score)

        return score  # size=(N, n_class, x.H/1, x.W/1)


class VGGNet(VGG):
    def __init__(self, pretrained=True, model='vgg16', requires_grad=True, remove_fc=True, show_params=False):
        super().__init__(make_layers(cfg[model]))
        self.ranges = ranges[model]

        if pretrained:
            exec("self.load_state_dict(models.%s(pretrained=True).state_dict())" % model)

        if not requires_grad:
            for param in super().parameters():
                param.requires_grad = False

        if remove_fc:  # delete redundant fully-connected layer params, can save memory
            del self.classifier

        if show_params:
            for name, param in self.named_parameters():
                print(name, param.size())

    def forward(self, x):
        output = {}

        # get the output of each maxpooling layer (5 maxpool in VGG net)
        for idx in range(len(self.ranges)):
            for layer in range(self.ranges[idx][0], self.ranges[idx][1]):
                x = self.features[layer](x)
            output["x%d"%(idx+1)] = x

        return output


ranges = {
    'vgg11': ((0, 3), (3, 6),  (6, 11),  (11, 16), (16, 21)),
    'vgg13': ((0, 5), (5, 10), (10, 15), (15, 20), (20, 25)),
    'vgg16': ((0, 5), (5, 10), (10, 17), (17, 24), (24, 31)),
    'vgg19': ((0, 5), (5, 10), (10, 19), (19, 28), (28, 37))
}

# cropped version from https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
cfg = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


