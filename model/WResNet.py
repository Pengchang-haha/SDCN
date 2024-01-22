import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import init


# WResNet
class WResNet(nn.Module):
    def __init__(self, weighted_average=True):
        super(WResNet, self).__init__()
        #layer1
        #self.conv1 = nn.Conv2d(3, 64, 3, padding=1,stride=2)
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        #layer2
        self.conv5 = nn.Conv2d(64, 128, 3, padding=1, stride=2)
        self.conv6 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv7 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv8 = nn.Conv2d(128, 128, 3, padding=1)
        # layer3
        self.conv9 = nn.Conv2d(128, 256, 3, padding=1, stride=2)
        self.conv10 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv11 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv12 = nn.Conv2d(256, 256, 3, padding=1)

        self.layer3conv3 = nn.Conv2d(256, 512, 3, padding=1, stride=2)
        self.layer3conv4 = nn.Conv2d(512, 512, 3, padding=1, stride=2)

        # layer4
        self.conv13 = nn.Conv2d(256, 256, 3, padding=1, stride=2)
        self.conv14 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv15 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv16 = nn.Conv2d(256, 256, 3, padding=1)
        # layer5
        self.conv17 = nn.Conv2d(256, 512, 3, padding=1, stride=2)
        self.conv18 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv19 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv20 = nn.Conv2d(512, 512, 3, padding=1)  #7*7
        
        # FC
        self.fc1_q = nn.Linear(512 * 3, 512)
        self.fc2_q = nn.Linear(512, 1)
        self.fc1_w = nn.Linear(512 * 3, 512)
        self.fc2_w = nn.Linear(512, 1)
        self.dropout = nn.Dropout()
        self.weighted_average = weighted_average



    def extract_features(self, x):
        x11 = self.conv1(x)#64*112*112
        x12 = F.relu(self.conv2(x11))+x11
        x13 = F.relu(self.conv3(x12))
        x14 = F.relu(self.conv4(x13)) + x11#64*112*112  64*32*32

        x21 = F.relu(self.conv5(x14)) #128*56*56
        x22 = F.relu(self.conv6(x21))+ x21
        x23 = F.relu(self.conv7(x22))
        x24 = F.relu(self.conv8(x23)) + x21#128*56*56   128*16*16

        x31 = F.relu(self.conv9(x24))  #256*28*28
        x32 = F.relu(self.conv10(x31))+ x31
        x33 = F.relu(self.conv11(x32))
        x34 = F.relu(self.conv12(x33)) + x31#256*28*28  256*8*8
       

        x41 = F.relu(self.conv13(x34))#256*14*14
        x42 = F.relu(self.conv14(x41))+ x41
        x43 = F.relu(self.conv15(x42))
        x44 = F.relu(self.conv16(x43)) + x41#256*14*14  256*4*4

        x51 = F.relu(self.conv17(x44))#512*7*7
        x52 = F.relu(self.conv18(x51))+ x51
        x53 = F.relu(self.conv19(x52))
        x54 = F.relu(self.conv20(x53)) + x51#512*7*7   512*2*2

        x6 = F.avg_pool2d(x54, 2)#512*1*1  x54是输入，2是核
        x6 = x6.view(-1, 512)#reshape

        return x6#,x11

  
    def forward(self, data):
        """
        :param data: distorted and reference patches of images
        :return: quality of images/patches
        """
        x, x_ref = data     # [batch_size, patch_num, 3, 32, 32]
        batch_size = x.size(0)
        n_patches = x.size(1)
        if self.weighted_average:
            q = torch.ones((batch_size, 1), device=x.device)
            #torch.ones(*size, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) → Tensor
            #返回创建size大小的维度，里面元素全部填充为1
        else:
            q = torch.ones((batch_size * n_patches, 1), device=x.device)

        for i in range(batch_size):
            # x[i] - [patch_num, 3,32,32]
            h = self.extract_features(x[i])  # [patch_num, 512] 79行定义了extract_features
            h_ref = self.extract_features(x_ref[i])  # [patch_num, 512]

            f_cat = torch.cat((h - h_ref, h, h_ref), 1)  # [patch_num, 512 * 3]

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
    
