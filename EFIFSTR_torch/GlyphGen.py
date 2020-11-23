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
import utils
import torch.nn.functional as F
from torch.utils.data import *
import easydict
import pickle
import time
import os

class Generator(nn.Module):
    def __init__(self, opt, device):
        super(Generator, self).__init__()
        self.fc_1 = nn.Linear(int(opt.img_h/2) * int(opt.img_w_max/2), 16 * 16)
        self.fc_2 = nn.Linear(int(opt.img_h/4) * int(opt.img_w_max/4), 8 * 8)
        self.fc_3 = nn.Linear(int(opt.img_h/8) * int(opt.img_w_max/4), 4 * 4)
        self.font_embedding = nn.Embedding(opt.num_fonts, 128)
        self.Deconv_1 = nn.ConvTranspose2d(opt.fmap_dim + 128, 128, kernel_size =[2,2], stride=[2,2] )
        self.Deconv_2 = nn.ConvTranspose2d(128, 64, kernel_size=[2,2], stride=[2,2]) #kernel size [3,3] make output size  [T, S, 5 , 5]
        self.Deconv_3 = nn.ConvTranspose2d((64 + 128), 32, kernel_size=[2,2], stride=[2,2]) # ref : Resnet encoder layer3
        self.Deconv_4 = nn.ConvTranspose2d((32 + 64), 16, kernel_size=[2,2], stride=[2,2])  # ref : Resnet encoder layer2
        self.Deconv_5 = nn.ConvTranspose2d((16 + 32), 1, kernel_size=[2,2], stride=[2,2])   # ref : Resnet encoder layer1
        self.ref_glyphs = torch.Tensor(np.load('./EFIFSTR/data/glyphs_ko_104.npy')).to(device)
        self.criterion = torch.nn.L1Loss(reduction='none').to(device)
        self.device=device

    def generate_glimpse(self, fmap, masks):
        _, fmap_c, fmap_h, fmap_w = fmap.shape
        mask = F.interpolate(masks, size = (fmap_h, fmap_w), mode='bilinear', align_corners=False)
        mask = mask.repeat(1, fmap_c, 1, 1)
        fmap = fmap.unsqueeze(1)  #[N, 1, c, 24, 80]
        fmap = fmap.repeat(1, self.seq_len, 1, 1, 1).reshape((self.batch_size * self.seq_len, fmap_c, fmap_h, fmap_w))
        glimpse = torch.mul(mask, fmap)
        
        #reshape
        glimpse = glimpse.reshape((self.batch_size * self.seq_len, fmap_c, fmap_h * fmap_w))
        
        return glimpse, fmap_c
    
    
    def forward(self, feature_map_list, masks, glimpses):
        
        self.batch_size,  self.seq_len,  self.height,  self.width = masks.shape 
        masks = masks.reshape((self.batch_size * self.seq_len, 1, self.height, self.width))
        
        glimpse_s1, fmap_s1_c = self.generate_glimpse(feature_map_list[0], masks)  # feature_map_list[0] shape  # 24 * 80
        glimpse_s2, fmap_s2_c = self.generate_glimpse(feature_map_list[1], masks)   # 12 * 40
        glimpse_s3, fmap_s3_c = self.generate_glimpse(feature_map_list[2], masks)   # 6 * 40
        
        fmap_last = feature_map_list[-1] # 6 * 40
        _, feature_c, feature_h, feature_w = fmap_last.shape       
        
#         ### fmap s3
#         mask_s3 = masks.repeat(1,fmap_s1_c, 1,1)
#         fmap_s3.unsqueeze_(1) #  after unsqueeze -> [N, 1, c, 6, 40]
#         fmap_s3 = fmap_s3.repeat(1, seq_len, 1, 1, 1).reshape((batch_size * seq_len, fmap_s3_c, fmap_s3_h, fmap_s3_w))
#         glimpse_s3 = torch.mul(mask_s3, fmap_s3)
        
#         ## fmap s2
#         mask_s2 = F.interpolate(masks, size = (fmap_s2_h, fmap_s2_w), mode='bilinear', align_corners=False)
#         fmap_s2.unsqueeze_(1)  #[N, 1, c, 12, 40]
#         fmap_s2 = fmap_s2.repeat(1, seq_len, 1, 1, 1).reshape((batch_size * seq_len, fmap_s2_c, fmap_s2_h, fmap_s2_w))
#         glimpse_s2 = torch.mul(mask_s2, fmap_s2)

        glimpse_1 = self.fc_1(glimpse_s1).reshape((self.batch_size * self.seq_len , fmap_s1_c, 16, 16))
        glimpse_2 = self.fc_2(glimpse_s2).reshape((self.batch_size * self.seq_len , fmap_s2_c, 8, 8))
        glimpse_3 = self.fc_3(glimpse_s3).reshape((self.batch_size * self.seq_len , fmap_s3_c, 4, 4))
        
        embedding_ids = torch.randint(low=0, high= 104, size=(self.batch_size * self.seq_len, 1)).to(self.device)
        font_embedded = self.font_embedding(embedding_ids).reshape((self.batch_size * self.seq_len, 128, 1, 1))
        
        # deconv stage
        glimpses_deconv = glimpses.reshape((self.batch_size * self.seq_len, feature_c, 1, 1))
        concat_deconv = torch.cat([glimpses_deconv, font_embedded], axis=1)
        
        d1 = self.Deconv_1(concat_deconv)
        d2 = self.Deconv_2(d1)
        d3 = self.Deconv_3(torch.cat([d2, glimpse_3], dim=1))
        d4 = self.Deconv_4(torch.cat([d3, glimpse_2], dim=1))
        d5 = self.Deconv_5(torch.cat([d4, glimpse_1], dim=1))
        d5 = torch.tanh(d5)
        
        glyph = d5.reshape((self.batch_size ,self.seq_len, 32 * 32)) 
        
        return glyph, embedding_ids

    def glyph_loss(self, glyphs, target, target_length, embedding_ids, opt):
        target_glyph_ids = target.reshape((opt.batch_size*(opt.max_length+1), 1)) + 2447 * embedding_ids 
        ref_target = self.ref_glyphs[target_glyph_ids.reshape(-1,)] 
        ref_target = ref_target.reshape((opt.batch_size, opt.max_length+1, 32*32))
#         l1_loss_ = F.l1_loss(glyphs, ref_target, reduction='none')
        l1_loss_ = self.criterion(glyphs, ref_target)
        l1_avg = torch.mean(l1_loss_, dim=2) # [N, max+seq+1]
        
        max_lengths = torch.Tensor(list(range(opt.max_length +1))).repeat((opt.batch_size, 1)).to(self.device)
        loss_mask = torch.lt(max_lengths, target_length.reshape(-1,1))
        masked_loss = torch.mul(l1_avg, loss_mask)
        row_losses = torch.sum(masked_loss, dim= 1)
        loss = torch.div(torch.sum(row_losses), opt.batch_size)
        generation_loss = loss * 0.5
        
        return generation_loss