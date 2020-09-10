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
import Pred_jamo_position
import importlib
importlib.reload(PositionEnhancement)
importlib.reload(Hybrid)

class model(nn.Module):
    def __init__(self, opt, device):
        super(model, self).__init__()
        self.opt = opt
        
        #Trans
        self.Trans = Trans.TPS_SpatialTransformerNetwork(F = opt.num_fiducial,
                                                  i_size = (opt.imgH, opt.imgW), 
                                                  i_r_size= (opt.imgH, opt.imgW), 
                                                  i_channel_num=opt.input_channel,
                                                        device = device)
        #Extract
        if self.opt.extract =='RCNN':
            self.Extract = self.Extract = Extract.RCNN_extractor(opt.input_channel, opt.output_channel)
        elif 'efficientnet' in self.opt.extract :
            self.Extract = Extract.EfficientNet(opt)
        elif 'resnet' in self.opt.extract:
            self.Extract = Extract.ResNet_FeatureExtractor(opt.input_channel, opt.output_channel)
        else:
            raise print('invalid extract model name!')
            
#         self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None,1)) # imgH/16-1   ->  1
            
        #  Position aware module
        self.PAM = PositionEnhancement.PositionAwareModule(opt.output_channel, opt.hidden_size, opt.output_channel, 2)
        
        self.PAttnM_bot = PositionEnhancement.AttnModule(opt, opt.hidden_size, opt.bot_n_cls, device)
        self.PAttnM_mid = PositionEnhancement.AttnModule(opt, opt.hidden_size, opt.mid_n_cls, device)
        self.PAttnM_top = PositionEnhancement.AttnModule(opt, opt.hidden_size, opt.top_n_cls, device)
       
        
        # Hybrid branch
        self.Hybrid_bot = Hybrid.HybridBranch(opt.output_channel, opt.batch_max_length+1, opt.bot_n_cls, device)
        self.Hybrid_mid = Hybrid.HybridBranch(opt.output_channel, opt.batch_max_length+1, opt.mid_n_cls, device)
        self.Hybrid_top = Hybrid.HybridBranch(opt.output_channel, opt.batch_max_length+1, opt.top_n_cls, device)
            
        
#         # Dynamically fusing module
        self.Dynamic_fuser_top = PositionEnhancement.DynamicallyFusingModule(opt.top_n_cls)
        self.Dynamic_fuser_mid = PositionEnhancement.DynamicallyFusingModule(opt.mid_n_cls)
        self.Dynamic_fuser_bot = PositionEnhancement.DynamicallyFusingModule(opt.bot_n_cls)
        

        
    def forward(self, input, text_bot, text_mid, text_top, is_train=True):
        #Trans stage
        input = self.Trans(input)
        
        #Extract stage
        visual_feature = self.Extract(input)
#         print(visual_feature)

        position_feature = self.PAM(visual_feature)
#         print('Position feature : ' ,position_feature)
        
        g_prime_bot = self.PAttnM_bot(position_feature.permute(0,2,3,1), visual_feature.permute(0,2,3,1))
        g_prime_mid = self.PAttnM_mid(position_feature.permute(0,2,3,1), visual_feature.permute(0,2,3,1))
        g_prime_top = self.PAttnM_top(position_feature.permute(0,2,3,1), visual_feature.permute(0,2,3,1))
        
#         print('G`bot : ', g_prime_bot)
#         print('G`mid : ', g_prime_mid)
#         print('G`top : ', g_prime_top)
        
        g_bot = self.Hybrid_bot(visual_feature, text_bot, is_train)
        g_mid = self.Hybrid_mid(visual_feature, text_mid, is_train)
        g_top = self.Hybrid_top(visual_feature, text_top, is_train)
        
#         print('G bot : ', g_bot)
#         print('G mid : ', g_mid)
#         print('G top : ', g_top)
        
        pred_bot = self.Dynamic_fuser_bot(g_bot, g_prime_bot)
        pred_mid = self.Dynamic_fuser_mid(g_mid, g_prime_mid)
        pred_top = self.Dynamic_fuser_top(g_top, g_prime_top)
        
        return pred_top, pred_mid, pred_bot
    
    
    
#         print('extract output : ', visual_feature.shape)
#         visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2))
#         print('average pool output : ', visual_feature.shape)
#         visual_feature = visual_feature.squeeze(3) #(batch_size, num_seq, vectors) ex) 192, 23, 512
# #         print(f'Extract stage shape : {visual_feature.shape}')
        
#         #Seq stage
#         contextual_feature = self.Seq(visual_feature) # same shape as previous stage ex) 192, 23, 512
# #         print(f'Seq stage shape : {contextual_feature.shape}')     
        
#         #Pred stage
#         g_bot = self.Pred_bot(contextual_feature.contiguous(), text_bot, is_train, batch_max_length = self.opt.batch_max_length)
#         _, g_bot_idx = g_bot.max(2) 
        
#         g_mid = self.Pred_mid(contextual_feature.contiguous(), text_mid, g_bot_idx, is_train, batch_max_length = self.opt.batch_max_length)
#         _, g_mid_idx = g_mid.max(2)
        
#         g_top = self.Pred_top(contextual_feature.contiguous(), text_top, g_mid_idx, g_bot_idx,  is_train, 
#                                  batch_max_length = self.opt.batch_max_length)

#         g_prime_bot = self.Position_attn_bot(contextual_feature.contiguous())
#         g_prime_mid = self.Position_attn_mid(contextual_feature.contiguous())
#         g_prime_top = self.Position_attn_top(contextual_feature.contiguous())
        
#         pred_bot = self.Dynamic_fuser_bot(g_bot, g_prime_bot)
#         pred_mid = self.Dynamic_fuser_mid(g_mid, g_prime_mid)
#         pred_top = self.Dynamic_fuser_top(g_top, g_prime_top)
        
        
#         return pred_top, pred_mid, pred_bot