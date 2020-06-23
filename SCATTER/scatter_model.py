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
import SCR
import CTC

class SCATTER(nn.Module):
    def __init__(self, opt, device):
        super(SCATTER, self).__init__()
        self.opt = opt
        
        #Trans
        self.Trans = Trans.TPS_SpatialTransformerNetwork(F = opt.num_fiducial, i_size = (opt.imgH, opt.imgW), 
                                                  i_r_size= (opt.imgH, opt.imgW), i_channel_num=opt.input_channel, device = device)
        
        #Extract
#         self.Extract = Extract.RCNN_extractor(opt.input_channel, opt.output_channel)
#         self.Extract = Extract.ResNet_FeatureExtractor(opt.input_channel, opt.output_channel)
        self.Extract = Extract.EfficientNet(opt)
        self.FeatureExtraction_output = opt.output_channel # (imgH/16 -1 )* 512
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None,1)) # imgH/16-1   ->  1
            
        # VISUAL FEATURES 
        self.VFR = VFR.Visual_Features_Refinement(kernel_size = (3,1), num_classes = opt.num_classes, 
                                              in_channels = self.FeatureExtraction_output, out_channels=1, stride=1)
        
        # CTC DECODER
#         self.CTC = CTC.CTC_decoder(opt.output_channel, opt.output_channel, opt.num_classes, opt, device)
            
        # Selective Contextual Refinement Block
        self.SCR_1 = SCR.Selective_Contextual_refinement_block(input_size = self.FeatureExtraction_output, 
                                                         hidden_size = int(self.FeatureExtraction_output/2),
                                                        output_size = self.FeatureExtraction_output,
                                                        num_classes = opt.num_classes, decoder_fix = False, device = device, 
                                                        batch_max_length = opt.batch_max_length)
        
        self.SCR_2 = SCR.Selective_Contextual_refinement_block(input_size = self.FeatureExtraction_output, 
                                                         hidden_size = int(self.FeatureExtraction_output/2),
                                                        output_size = self.FeatureExtraction_output,
                                                        num_classes = opt.num_classes, decoder_fix = False, device = device,
                                                        batch_max_length = opt.batch_max_length)
        
        self.SCR_3 = SCR.Selective_Contextual_refinement_block(input_size = self.FeatureExtraction_output, 
                                                         hidden_size = int(self.FeatureExtraction_output/2),
                                                        output_size = self.FeatureExtraction_output,
                                                        num_classes = opt.num_classes, decoder_fix = True, device = device,
                                                        batch_max_length = opt.batch_max_length)
  
    def forward(self, input, text, is_train):
        # Trans stage
        input = self.Trans(input)
        
        # Extract stage
        visual_feature = self.Extract(input) # visual_feature.shape) # (192, 512, 1 , 23)
        
        # Visual Feature Refinement
        visual_refined = self.VFR(visual_feature) # visual_ refined output Size([192, 23, 512])
        
        # CTC DECODER
#         ctc_prob  = self.CTC(visual_refined, text, opt)
        
        #Selective Contextual Refinement
        scr_probs_1, H = self.SCR_1(visual_feature.permute(0, 3, 1, 2).squeeze(3), text, is_train)
        scr_probs_2, H = self.SCR_2(H, text, is_train)
        scr_probs_3, H = self.SCR_3(H, text, is_train)
        

        return [scr_probs_1, scr_probs_2, scr_probs_3]
#         return visual_refined, _
    