import os
import random
import torch
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.optim as optim
import torch.utils.data
import numpy as np
import six
import math
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.data import *


class Visual_Features_Refinement(torch.nn.Module):
    def __init__( self, kernel_size, num_classes, in_channels=512, out_channels=1, stride=1):
        super(Visual_Features_Refinement, self).__init__()
        if len(kernel_size) ==1:
            height_kernel = kernel_size
            width_kernel = kernel_size
        else :
            height_kernel = kernel_size[0]
            width_kernel = kernel_size[1]
            
        height_padding = self.pad_calc(height_kernel)
        width_padding= self.pad_calc(width_kernel)
        self.conv_3x1 = torch.nn.Conv2d(in_channels = in_channels, out_channels = out_channels, stride= stride, 
                                        kernel_size = (height_kernel, width_kernel), padding= (height_padding, width_padding))
        self.sigmoid = torch.sigmoid
        self.refiner = torch.nn.Linear(in_channels, num_classes)
        
    def forward(self, feature_output):
        assert feature_output[0][0].shape[0] % 2==1 #assume that feature_output height, width each are odd number shape
        assert feature_output[0][0].shape[1] % 2==1 # input_image *1/8 must be odd shape
        
        # text_recognition module
        conv_output = self.conv_3x1(feature_output)
        attention_mask = self.sigmoid(conv_output)
        assert attention_mask[0][0].shape == feature_output[0][0].shape # to enable broadcast operation
        
        attention_feature = feature_output * attention_mask
        attention_feature = attention_feature.permute(0, 3, 1, 2).squeeze(3)
#         print(attention_feature.shape)
        
        #visual refining
        feature_refined = self.refiner(attention_feature)
 
        return attention_feature
    
    def pad_calc(self, kernel_size):
        if kernel_size == 1:
            return 0
        elif kernel_size == 3:
            return 1
        elif kernel_size == 5:
            return 2
        
    