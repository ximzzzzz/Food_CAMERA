import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.optim as optim
import torch.utils.data

import re
import six
import math
import torchvision.transforms as transforms

import utils
from utils import *
import Trans
import Extract
from Seq import BidirectionalLSTM
import Seq
import Pred_jamo_position
import importlib
importlib.reload(Pred_jamo_position)


class STR(nn.Module):
    def __init__(self, opt, device):
        super(STR, self).__init__()
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
        else:
            raise print('invalid extract model name!')
            
#         self.Extract = Extract.ResNet_FeatureExtractor(opt.input_channel, opt.output_channel)
        self.FeatureExtraction_output = opt.output_channel # (imgH/16 -1 )* 512
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None,1)) # imgH/16-1   ->  1

            
        # Sequence
        self.Seq = nn.Sequential(
            BidirectionalLSTM(self.FeatureExtraction_output, opt.hidden_size,  opt.hidden_size),
            BidirectionalLSTM(opt.hidden_size, opt.hidden_size, opt.hidden_size))
        self.Seq_output = opt.hidden_size
        
        
        #Pred (Hybrid branch)

        self.Pred_bot = Pred_jamo_position.Attention(self.Seq_output, opt.hidden_size, opt.bottom_n_cls, device=device)
        self.Pred_mid = Pred_jamo_position.Attention_mid(self.Seq_output, opt.hidden_size, opt.middle_n_cls, opt.bottom_n_cls,  device=device)
        self.Pred_top = Pred_jamo_position.Attention_top(self.Seq_output, opt.hidden_size, opt.top_n_cls, opt.middle_n_cls, opt.bottom_n_cls, device=device)
            
        # Position enhancement module
        
        self.Position_attn_top = Pred_jamo_position.PositionAttnModule(opt, opt.top_n_cls, device)
        self.Position_attn_mid = Pred_jamo_position.PositionAttnModule(opt, opt.middle_n_cls, device)
        self.Position_attn_bot = Pred_jamo_position.PositionAttnModule(opt, opt.bottom_n_cls, device)
        
        # Dynamically fusing module
        
        self.Dynamic_fuser_top = Pred_jamo_position.DynamicallyFusingModule(opt.top_n_cls)
        self.Dynamic_fuser_mid = Pred_jamo_position.DynamicallyFusingModule(opt.middle_n_cls)
        self.Dynamic_fuser_bot = Pred_jamo_position.DynamicallyFusingModule(opt.bottom_n_cls)
        

        
    def forward(self, input, text_top, text_mid, text_bot, is_train=True):
        #Trans stage
        input = self.Trans(input)
        
        #Extract stage
        visual_feature = self.Extract(input)
#         print('extract output : ', visual_feature.shape)
        visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2))
#         print('average pool output : ', visual_feature.shape)
        visual_feature = visual_feature.squeeze(3) #(batch_size, num_seq, vectors) ex) 192, 23, 512
#         print(f'Extract stage shape : {visual_feature.shape}')
        
        #Seq stage
        contextual_feature = self.Seq(visual_feature) # same shape as previous stage ex) 192, 23, 512
#         print(f'Seq stage shape : {contextual_feature.shape}')     
        
        #Pred stage
        g_bot = self.Pred_bot(contextual_feature.contiguous(), text_bot, is_train, batch_max_length = self.opt.batch_max_length)
        _, g_bot_idx = g_bot.max(2) 
        
        g_mid = self.Pred_mid(contextual_feature.contiguous(), text_mid, g_bot_idx, is_train, batch_max_length = self.opt.batch_max_length)
        _, g_mid_idx = g_mid.max(2)
        
        g_top = self.Pred_top(contextual_feature.contiguous(), text_top, g_mid_idx, g_bot_idx,  is_train, 
                                 batch_max_length = self.opt.batch_max_length)

        g_prime_bot = self.Position_attn_bot(contextual_feature.contiguous())
        g_prime_mid = self.Position_attn_mid(contextual_feature.contiguous())
        g_prime_top = self.Position_attn_top(contextual_feature.contiguous())
        
        pred_bot = self.Dynamic_fuser_bot(g_bot, g_prime_bot)
        pred_mid = self.Dynamic_fuser_mid(g_mid, g_prime_mid)
        pred_top = self.Dynamic_fuser_top(g_top, g_prime_top)
        
        
        return pred_top, pred_mid, pred_bot