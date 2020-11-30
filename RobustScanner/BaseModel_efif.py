import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.optim as optim
import torch.utils.data
import sys
sys.path.append('../Whatiswrong')

import re
import six
import math
import torchvision.transforms as transforms

import utils
from utils import *
import Trans
import Extract
import PositionEnhancement
import Hybrid
import importlib
from efficientnet_pytorch import EfficientNet
import torchvision.models as models

importlib.reload(PositionEnhancement)
importlib.reload(Hybrid)

class model(nn.Module):
    def __init__(self, opt, device):
        super(model, self).__init__()
        self.opt = opt
        
        #Trans
        if self.opt.trans:
            self.Trans = Trans.TPS_SpatialTransformerNetwork(F = opt.num_fiducial,
                                                  i_size = (opt.imgH, opt.imgW), 
                                                  i_r_size= (opt.imgH, opt.imgW), 
                                                  i_channel_num=opt.input_channel,
                                                        device = device)
        #Extract
        if self.opt.extract =='RCNN':
            self.Extract = self.Extract = Extract.RCNN_extractor(opt.input_channel, opt.output_channel)
        elif 'efficientnet' in self.opt.extract :    
            self.Extract = EfficientNet.from_name(opt.extract)
#             self.Nin = torch.nn.Conv2d(1536, opt.output_channel, 1) # b3
            self.Nin = torch.nn.Conv2d(2304, opt.output_channel, 1) # b6
        elif 'resnet' in self.opt.extract:
            self.Extract = Extract.ResNet_FeatureExtractor(opt.input_channel, opt.output_channel)
        elif 'densenet' in self.opt.extract:
            self.Extract = torch.hub.load('pytorch/vision:v0.6.0', 'densenet161', pretrained=False)
#             self.Nin = torch.nn.Conv2d(1024, opt.output_channel, 1) # 121
            self.Nin = torch.nn.Conv2d(2208, opt.output_channel, 1) # 161
        elif 'resnext' in self.opt.extract:
            self.Extract = nn.Sequential(*list(models.resnext101_32x8d().children())[:-2])
            self.Nin = torch.nn.Conv2d(2048, opt.output_channel, 1)
            
        else:
            raise print('invalid extract model name!')
            
#         self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None,1)) # imgH/16-1   ->  1
   
        #  Position aware module
        self.PAM = PositionEnhancement.PositionAwareModule(opt.output_channel, opt.hidden_size, opt.output_channel, 2)
        
        self.PAttnM = PositionEnhancement.AttnModule(opt, opt.hidden_size, opt.bot_n_cls, device)

        # Hybrid branch
        self.Hybrid = Hybrid.HybridBranch(opt.output_channel, opt.batch_max_length+1, opt.bot_n_cls, device)
            
#         # Dynamically fusing module
        self.Dynamic_fuser = PositionEnhancement.DynamicallyFusingModule(opt.top_n_cls)
        

    def forward(self, input, text, is_train=False):
        
        #Trans stage
        if self.opt.trans:
            input = self.Trans(input)
        
        #Extract stage
        if 'efficientnet' in self.opt.extract:
            visual_feature = self.Extract.extract_features(input)
            visual_feature = self.Nin(visual_feature)
            
        elif 'densenet' in self.opt.extract:
            visual_feature = self.Extract.features(input)
            visual_feature = self.Nin(visual_feature)
            
        elif 'resnext' in self.opt.extract:
            visual_feature = self.Extract(input)
            visual_feature = self.Nin(visual_feature)
            
        else:
            visual_feature = self.Extract(input)


        position_feature = self.PAM(visual_feature)
#         print('Position feature : ' ,position_feature)
        
        g_prime, context = self.PAttnM(position_feature.permute(0,2,3,1), visual_feature.permute(0,2,3,1))
        
#         print('G`bot : ', g_prime_bot)

        g = self.Hybrid(visual_feature, text, is_train)
        
#         print('G bot : ', g_bot)
#         print('G mid : ', g_mid)
#         print('G top : ', g_top)
        
        pred = self.Dynamic_fuser(g, g_prime)

        return pred
    
