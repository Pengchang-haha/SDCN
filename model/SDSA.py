#2个基础QKV侧枝再第三个块后

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import init
from model.Nonlocal1 import *
from model.deformable import *
from torchvision.ops.deform_conv import DeformConv2d

# WResNet
class SDSA(nn.Module):
    def __init__(self, weighted_average=True):
        super(SDSA, self).__init__()
        #layer1
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1, stride=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)#32*32*64
        #layer2
        self.conv5 = nn.Conv2d(64, 128, 3, padding=1, stride=2)
        self.conv6 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv7 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv8 = nn.Conv2d(128, 128, 3, padding=1)#16*16*28
        # layer3
        self.conv9 = nn.Conv2d(128, 256, 3, padding=1, stride=2)
        self.conv10 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv11 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv12 = nn.Conv2d(256, 256, 3, padding=1)#8*8*256
        # 2个3*3conv尺寸加平均池化+残差QKV，
        self.layer3conv3 = nn.Conv2d(256, 256, 3, padding=1, stride=2)#4*4*512
        self.layer3conv4 = nn.Conv2d(256, 512, 3, padding=1, stride=2)#输出尺寸2*2*512
        self.avgpool = nn.AvgPool2d(2, 2)  # k=2,s=2 1*1*512
        self.Nonlocal11 = NONLocalBlock2D(512, 256, sub_sample=False, bn_layer=False)
        self.Nonlocal12 = NONLocalBlock2D(512, 256, sub_sample=False, bn_layer=False)

        # layer4
        self.conv13 = nn.Conv2d(256, 256, 3, padding=1, stride=2)
        self.conv14 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv15 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv16 = nn.Conv2d(256, 256, 3, padding=1)#4*4*256

        # 可变形卷积
        # d1
        self.conv_offset1 = nn.Conv2d(256, 2 * 3 * 3, 3, 1, 1)
        self.deform11 = DeformConv2d(256, 256, 3, 1, 1)
        self.deform12 = DeformConv2d(256, 256, 3, 1, 1)
        # d2
        self.conv_offset2 = nn.Conv2d(256, 2 * 3 * 3, 3, 1, 1)
        self.deform21 = DeformConv2d(256, 256, 3, 1, 1)
        self.deform22 = DeformConv2d(256, 256, 3, 1, 1)
        # d3
       # self.conv_offset3 = nn.Conv2d(256, 2 * 3 * 3, 3, 1, 1)
       # self.deform31 = DeformConv2d(256, 256, 3, 1, 1)
       # self.deform32 = DeformConv2d(256, 256, 3, 1, 1)

        # layer5
        self.conv17 = nn.Conv2d(256, 512, 3, padding=1, stride=2)
        self.conv18 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv19 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv20 = nn.Conv2d(512, 512, 3, padding=1)#2*2*512
        # QKV
        self.Nonlocal21 = NONLocalBlock2D(512, 256, sub_sample=False, bn_layer=False)
        self.Nonlocal22 = NONLocalBlock2D(512, 256, sub_sample=False, bn_layer=False)

        # FC
        self.fc1_q = nn.Linear(512 * 3, 512)
        self.fc2_q = nn.Linear(512, 1)
        self.fc1_w = nn.Linear(512 * 3, 512)
        self.fc2_w = nn.Linear(512, 1)
        self.dropout = nn.Dropout()
        self.weighted_average = weighted_average



    def extract_features(self, x):    #input:32*32*3
        x11 = self.conv1(x)
        x12 = F.relu(self.conv2(x11))+x11
        x13 = F.relu(self.conv3(x12))
        x14 = F.relu(self.conv4(x13)) + x11 #32*32*64

        x21 = F.relu(self.conv5(x14))
        x22 = F.relu(self.conv6(x21))+ x21
        x23 = F.relu(self.conv7(x22))
        x24 = F.relu(self.conv8(x23)) + x21 #16*16*128

        x31 = F.relu(self.conv9(x24))
        x32 = F.relu(self.conv10(x31))+ x31
        x33 = F.relu(self.conv11(x32))
        x34 = F.relu(self.conv12(x33)) + x31 #8*8*256

        #x11 = F.relu(self.layer3conv3(x34))  #4*4*512
        #x11 = self.layer3conv4(x11)  # 1*1conv尺寸2*2*512
        #x11 = F.avg_pool2d(x11, 2)  # 全局平均池化，然后输入进QKV尺寸1*1*512


        return x34#,x11
    #侧枝QKV
    def extract_featuresaQKV(self, x):

        x11 = F.relu(self.layer3conv3(x))  #4*4*512
        x11 = self.layer3conv4(x11)  # 1*1conv尺寸2*2*512
        x11 = F.avg_pool2d(x11, 2)  # 全局平均池化，然后输入进QKV尺寸1*1*512

        return x11

    def extract_featuresa(self, x):
        x41 = F.relu(self.conv13(x))
        x42 = F.relu(self.conv14(x41)) + x41
        x43 = F.relu(self.conv15(x42))
        x44 = F.relu(self.conv16(x43)) + x41  # 4*4*512

        x51 = F.relu(self.conv17(x44))
        x52 = F.relu(self.conv18(x51))+ x51
        x53 = F.relu(self.conv19(x52))
        x54 = F.relu(self.conv20(x53)) + x51 #2*2*512

        x6 = F.avg_pool2d(x54, 2)# 全局平均池化，然后输入进QKV尺寸1*1
        res = x6.view(-1, 512)

        return res

    def forward(self, data):
        """
        :param data: distorted and reference patches of images
        :return: quality of images/patches
        """
        x, x_ref = data
        batch_size = x.size(0)
        n_patches = x.size(1)
        if self.weighted_average:
            q = torch.ones((batch_size, 1), device=x.device)
        else:
            q = torch.ones((batch_size * n_patches, 1), device=x.device)

        for i in range(batch_size):

            h = self.extract_features(x[i])  # h: [2 256 14 14]
            h_ref = self.extract_features(x_ref[i])
            #侧枝QKV
            h_x11=self.extract_featuresaQKV(h)
            h_ref_x11=self.extract_featuresaQKV(h_ref)

            # 第一个可变形卷积
            h_offset1 = self.conv_offset1(h_ref)
            h_deform = self.deform11(h, h_offset1)  # ,offset  (h,h_offset)
            h_deform_ref = self.deform12(h_ref, h_offset1)  # ,offset   (h_ref,h_offset)

            # 第二个可变形卷积
            h_offset2 = self.conv_offset2(h_deform_ref)
            h_deform = self.deform21(h_deform, h_offset2)  # ,offset  (h,h_offset)
            h_deform_ref = self.deform22(h_deform_ref, h_offset2)  # ,offset   (h_ref,h_offset)
            #print(h_deform)

            # 第三个可变形卷积
            #h_offset3 = self.conv_offset3(h_deform_ref)
            #h_deform = self.deform31(h_deform, h_offset3)  # ,offset  (h,h_offset) deform输出256，和x11的512不统一
            #h_deform_ref = self.deform32(h_deform_ref, h_offset3)  # ,offset   (h_ref,h_offset)

            # 主干QKV的输入
            h_deform1 = self.extract_featuresa(h_deform)  # 2*256*14*14 -> 18*512
            h_deform_ref1 = self.extract_featuresa(h_deform_ref)
            # 主干上的dis_QKV
            h_deform1 = h_deform1.unsqueeze(0)
            h_deform1 = h_deform1.transpose(1, 2)
            h_deform1 = h_deform1.unsqueeze(3)  # [1, 512, 18, 1]
            h_deform1 = self.Nonlocal21(h_deform1)
            # 主干上的ref_QKV
            h_deform_ref1 = h_deform_ref1.unsqueeze(0)
            h_deform_ref1 = h_deform_ref1.transpose(1, 2)
            h_deform_ref1 = h_deform_ref1.unsqueeze(3)
            h_deform_ref1 = self.Nonlocal22(h_deform_ref1)

            h_deform1 = h_deform1.squeeze(0)
            h_deform1 = h_deform1.transpose(0, 1)
            h_deform1 = h_deform1.squeeze(2)
            h_deform_ref1 = h_deform_ref1.squeeze(0)
            h_deform_ref1 = h_deform_ref1.transpose(0, 1)
            h_deform_ref1 = h_deform_ref1.squeeze(2)
            

            # 残差上的dis_QKV
            h_deform2 = h_x11.view(-1, 512)  # h_x11（2,512,3,3） 和 h_deform1（18,512） h_x11.view(-1, 512)
            h_deform2 = h_deform2.unsqueeze(0)
            h_deform2 = h_deform2.transpose(1, 2)
            h_deform2 = h_deform2.unsqueeze(3)  # [1, 512, 2, 1, 3, 3] 多出一个维度？
            h_deform2 = self.Nonlocal11(h_deform2)
            # 残差上的ref_QKV
            h_deform_ref2 = h_ref_x11.view(-1, 512)
            h_deform_ref2 = h_deform_ref2.unsqueeze(0)
            h_deform_ref2 = h_deform_ref2.transpose(1, 2)
            h_deform_ref2 = h_deform_ref2.unsqueeze(3)
            h_deform_ref2 = self.Nonlocal12(h_deform_ref2)

            h_deform2 = h_deform2.squeeze(0)
            h_deform2 = h_deform2.transpose(0, 1)
            h_deform2 = h_deform2.squeeze(2)
            h_deform_ref2 = h_deform_ref2.squeeze(0)
            h_deform_ref2 = h_deform_ref2.transpose(0, 1)
            h_deform_ref2 = h_deform_ref2.squeeze(2)
            #两个QKV的输出先相加，对应相加再按维度拼接
            h_deform3=h_deform1+h_deform2
            h_deform_ref3=h_deform_ref1+h_deform_ref2
            f_cat = torch.cat((h_deform3 - h_deform_ref3, h_deform3, h_deform_ref3), 1)  # [patch_num, 512 * 3]

            f = F.relu(self.fc1_q(f_cat))     # [patch_num, 512*3] -> [patch_num, 512]
            f = self.dropout(f)
            f = self.fc2_q(f)   # [patch_num, 512] -> [patch_num, 1]

            if self.weighted_average:
                w = F.relu(self.fc1_w(f_cat))  # [patch_num, 512*3] -> [patch_num, 512]
                w = self.dropout(w)
                w = F.relu(self.fc2_w(w)) + 0.000001    # [patch_num, 512] -> [patch_num, 1]
                q[i] = torch.sum(f * w) / torch.sum(w)  # weighted averaging
            else:
                q[i*n_patches:(i+1)*n_patches] = h

        return q

