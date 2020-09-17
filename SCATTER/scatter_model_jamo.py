import json
import pandas as pd
import os
import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.nn.init as init
from torch.utils.data import *
import numpy as np
import time
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import easydict
import sys
import re
import six
import math
import torchvision.transforms as transforms
from jamo import h2j, j2hcj, j2h

sys.path.append('./Whatiswrong')
sys.path.append('./Scatter')
import scatter_utils
import utils
import Trans
import Extract
import VFR
import SCR_jamo
import CTC_jamo
import importlib
importlib.reload(VFR)
importlib.reload(SCR_jamo)
importlib.reload(CTC_jamo)
importlib.reload(Extract)
importlib.reload(utils)


class SCATTER(nn.Module):
    def __init__(self, opt, device):
        super(SCATTER, self).__init__()
        self.opt = opt
        
        #Trans
        self.Trans = Trans.TPS_SpatialTransformerNetwork(F = opt.num_fiducial, i_size = (opt.imgH, opt.imgW), 
                                                  i_r_size= (opt.imgH, opt.imgW), i_channel_num=opt.input_channel, device = device)
        
        #Extract
        if self.opt.extract =='RCNN':
            self.Extract = Extract.RCNN_extractor(opt.input_channel, opt.output_channel)
        elif 'efficientnet' in self.opt.extract :
            self.Extract = Extract.EfficientNet(opt)
        elif 'resnet' in self.opt.extract:
            self.Extract = Extract.ResNet_FeatureExtractor(opt.input_channel, opt.output_channel)
        else:
            raise print('invalid extract model name!')
        #         self.Extract = Extract.ResNet_FeatureExtractor(opt.input_channel, opt.output_channel)
        self.FeatureExtraction_output = opt.output_channel # (imgH/16 -1 )* 512
#         self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None,1)) # imgH/16-1   ->  1
            
        # VISUAL FEATURES REFINE
        self.VFR = VFR.Visual_Features_Refinement(kernel_size = (3,1), in_channels = self.FeatureExtraction_output, out_channels=1, stride=1)
        
        # CTC DECODER
        self.CTC = CTC_jamo.CTC_decoder(opt.output_channel, opt.output_channel, opt.top_n_cls, opt.mid_n_cls, opt.bot_n_cls, device)
            
        # Selective Contextual Refinement Block
        
        self.SCR_bot_1 = SCR_jamo.Selective_Contextual_refinement_block(input_size = self.FeatureExtraction_output, 
                                                         hidden_size = int(self.FeatureExtraction_output/2),
                                                        output_size = self.FeatureExtraction_output,
                                                        vertical_num_classes = [opt.bot_n_cls], decoder_fix = False, device = device,
                                                        batch_max_length = opt.batch_max_length)
               
        self.SCR_bot_2 = SCR_jamo.Selective_Contextual_refinement_block(input_size = self.FeatureExtraction_output, 
                                                         hidden_size = int(self.FeatureExtraction_output/2),
                                                        output_size = self.FeatureExtraction_output,
                                                        vertical_num_classes = [opt.bot_n_cls], decoder_fix = False, device = device,
                                                        batch_max_length = opt.batch_max_length)        
        
        self.SCR_bot_3 = SCR_jamo.Selective_Contextual_refinement_block(input_size = self.FeatureExtraction_output, 
                                                         hidden_size = int(self.FeatureExtraction_output/2),
                                                        output_size = self.FeatureExtraction_output,
                                                        vertical_num_classes = [opt.bot_n_cls], decoder_fix = True, device = device,
                                                        batch_max_length = opt.batch_max_length)       
        
        self.SCR_mid_1 = SCR_jamo.Selective_Contextual_refinement_block(input_size = self.FeatureExtraction_output, 
                                                         hidden_size = int(self.FeatureExtraction_output/2),
                                                        output_size = self.FeatureExtraction_output,
                                                        vertical_num_classes = [opt.bot_n_cls, opt.mid_n_cls],
                                                        decoder_fix = False, device = device,
                                                        batch_max_length = opt.batch_max_length)
        
        self.SCR_mid_2 = SCR_jamo.Selective_Contextual_refinement_block(input_size = self.FeatureExtraction_output, 
                                                         hidden_size = int(self.FeatureExtraction_output/2),
                                                        output_size = self.FeatureExtraction_output,
                                                        vertical_num_classes = [opt.bot_n_cls, opt.mid_n_cls],
                                                        decoder_fix = False, device = device,
                                                        batch_max_length = opt.batch_max_length)        
        
        self.SCR_mid_3 = SCR_jamo.Selective_Contextual_refinement_block(input_size = self.FeatureExtraction_output, 
                                                         hidden_size = int(self.FeatureExtraction_output/2),
                                                        output_size = self.FeatureExtraction_output,
                                                        vertical_num_classes = [opt.bot_n_cls, opt.mid_n_cls],
                                                        decoder_fix = True, device = device,
                                                        batch_max_length = opt.batch_max_length)    
        
        self.SCR_top_1 = SCR_jamo.Selective_Contextual_refinement_block(input_size = self.FeatureExtraction_output, 
                                                         hidden_size = int(self.FeatureExtraction_output/2),
                                                        output_size = self.FeatureExtraction_output,
                                                        vertical_num_classes = [opt.bot_n_cls, opt.mid_n_cls, opt.top_n_cls], 
                                                        decoder_fix = False, device = device, 
                                                        batch_max_length = opt.batch_max_length)
        
        self.SCR_top_2 = SCR_jamo.Selective_Contextual_refinement_block(input_size = self.FeatureExtraction_output, 
                                                         hidden_size = int(self.FeatureExtraction_output/2),
                                                        output_size = self.FeatureExtraction_output,
                                                        vertical_num_classes = [opt.bot_n_cls, opt.mid_n_cls, opt.top_n_cls],
                                                        decoder_fix = False, device = device,
                                                        batch_max_length = opt.batch_max_length)
        
        self.SCR_top_3 = SCR_jamo.Selective_Contextual_refinement_block(input_size = self.FeatureExtraction_output, 
                                                         hidden_size = int(self.FeatureExtraction_output/2),
                                                        output_size = self.FeatureExtraction_output,
                                                        vertical_num_classes = [opt.bot_n_cls, opt.mid_n_cls, opt.top_n_cls],
                                                        decoder_fix = True, device = device,
                                                        batch_max_length = opt.batch_max_length)

#         self.SCR = SCR.SCR_Blocks(input_size = self.FeatureExtraction_output, 
#                                                          hidden_size = int(self.FeatureExtraction_output/2),
#                                                         output_size = self.FeatureExtraction_output,
#                                                         num_classes = opt.num_classes, device = device,
#                                                         batch_max_length = opt.batch_max_length, 
#                                                         n_blocks=opt.scr_n_blocks)
  
    def forward(self, input, text_top, text_mid, text_bot, is_train):
        # Trans stage
        input = self.Trans(input)
        
        # Extract stage
        visual_feature = self.Extract(input) # visual_feature.shape) # (192, 512, 1 , 23)
        
        # Visual Feature Refinement
        visual_refined = self.VFR(visual_feature) # visual_ refined output Size([192, 23, 512])
        
        top_prob_list = []
        mid_prob_list = []
        bot_prob_list = []
        
        # CTC DECODER
        top_prob, mid_prob, bot_prob  = self.CTC(visual_refined)
        
        top_prob_list.append(top_prob)
        mid_prob_list.append(mid_prob)
        bot_prob_list.append(bot_prob)
        
        #Selective Contextual Refinement
        visual_feature_perm = visual_feature.permute(0, 3, 1, 2).squeeze(3)
#         scr_probs_1, H = self.SCR_1(visual_feature.permute(0, 3, 1, 2).squeeze(3), text, is_train)
#         scr_probs_2, H = self.SCR_2(H, text, is_train)
#         scr_probs_3, H = self.SCR_3(H, text, is_train)

        bot_H1, bot_prob_list = self.SCR_bot_1(visual_feature_perm, visual_feature_perm, bot_prob_list, [text_bot], is_train)
        _, bot_prob_idx = bot_prob_list[-1].max(2)
        mid_H1, mid_prob_list = self.SCR_mid_1(visual_feature_perm, visual_feature_perm, mid_prob_list, [bot_prob_idx, text_mid], is_train)
        _, mid_prob_idx = mid_prob_list[-1].max(2)
        top_H1, top_prob_list = self.SCR_top_1(visual_feature_perm, visual_feature_perm, top_prob_list, [bot_prob_idx, mid_prob_idx, text_top], is_train)
        
        bot_H2, bot_prob_list = self.SCR_bot_2(bot_H1, visual_feature_perm, bot_prob_list, [text_bot], is_train)
        _, bot_prob_idx = bot_prob_list[-1].max(2)
        mid_H2, mid_prob_list = self.SCR_mid_2(mid_H1, visual_feature_perm, mid_prob_list, [bot_prob_idx, text_mid], is_train)
        _, mid_prob_idx = mid_prob_list[-1].max(2)
        top_H2, top_prob_list = self.SCR_top_2(top_H1, visual_feature_perm, top_prob_list, [bot_prob_idx, mid_prob_idx, text_top], is_train)        
        
            
        bot_prob_list = self.SCR_bot_3(bot_H2, visual_feature_perm, bot_prob_list, [text_bot], is_train)    
        _, bot_prob_idx = bot_prob_list[-1].max(2)
        mid_prob_list = self.SCR_mid_3(mid_H2, visual_feature_perm, mid_prob_list, [bot_prob_idx, text_mid], is_train)
        _, mid_prob_idx = mid_prob_list[-1].max(2)
        top_prob_list = self.SCR_top_3(top_H2, visual_feature_perm, top_prob_list, [bot_prob_idx, mid_prob_idx, text_top], is_train)

        
        return top_prob_list, mid_prob_list, bot_prob_list
    