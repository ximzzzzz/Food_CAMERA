import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.optim as optim
import numpy as np
import os
import sys
import cv2
from PIL import Image
import easydict
sys.path.append('./Whatiswrong')
sys.path.append('./Nchar_clf')
import Nchar_utils
import Extract
import utils
import torch.nn.functional as F
from torch.utils.data import *
import easydict
import torchvision
import pickle
import time
import os

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class ResnetBlock(nn.Module):

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResnetBlock, self).__init__()
        self.conv1 = conv1x1(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class Resnet_encoder(nn.Module):
    def __init__(self, opt, n_group=1):
        super(Resnet_encoder, self).__init__()
        self.n_group= n_group
        self.enc_dim = opt.enc_dim
        
        in_channels=3
        self.layer0 = nn.Sequential(nn.Conv2d(in_channels, 32, kernel_size=(3,3), stride=1, padding=1, bias=False),
                                   nn.BatchNorm2d(32),
                                   nn.ReLU(inplace=True))
        self.inplanes = 32
        self.layer1 = self._make_layer(32, 3, [2,2])  
        self.layer2 = self._make_layer(64, 4, [2,2])  
        self.layer3 = self._make_layer(128, 6, [2,1]) 
        self.layer4 = self._make_layer(256, 6, [1,1])
        self.layer5 = self._make_layer(512, 3, [1,1])
        
        self.rnn = nn.LSTM(512, int(self.enc_dim/2), num_layers=2, bidirectional=True, batch_first=True)
        self.out_planes = 2 * 256

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def _make_layer(self, planes, blocks, stride):
        downsample = None
        if stride !=[1,1] or self.inplanes != planes:
            downsample = nn.Sequential(conv1x1(self.inplanes, planes, stride),
                                      nn.BatchNorm2d(planes))
            
        layers = []
        layers.append(ResnetBlock(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(ResnetBlock(self.inplanes, planes))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x0 = self.layer0(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x5 = self.layer5(x4)
        feature_map = x5
        feature_map_list = [x1, x2, x3, x4, x5]
        
        batch_size, channels, feature_h, feature_w = feature_map.shape
        cnn_feat = F.max_pool2d(feature_map, (feature_h, 1))
        cnn_feat = cnn_feat.permute(0, 3, 1, 2).squeeze(3)
        
        self.rnn.flatten_parameters()
        _, holistic_feature = self.rnn(cnn_feat)
        
        # merge bidirection to uni-direction
        hidden_state, cell_state = holistic_feature    #hidden_state (num_layers * num_direction, batch_size, enc_dim / 2 ) 
        hidden_state = hidden_state.detach().transpose(0,1).contiguous().view(batch_size, -1, self.enc_dim ).transpose(0,1).contiguous()
        cell_state = cell_state.detach().transpose(0,1).contiguous().view(batch_size, -1, self.enc_dim).transpose(0,1).contiguous()
        
        return feature_map_list, (hidden_state, cell_state)
                   