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
import utils
import torch.nn.functional as F
from torch.utils.data import *
import pickle
import time
import os

#             0 features shape : torch.Size([5, 64, 64, 256])
#             1 features shape : torch.Size([5, 128, 32, 128])
#             2 features shape : torch.Size([5, 256, 16, 64])
#             3 features shape : torch.Size([5, 512, 8, 65])
#             4 features shape : torch.Size([5, 512, 3, 65])

class Generator(nn.Module):
    def __init__(self, opt, device):
        super(Generator, self).__init__()
        self.fc_1 = nn.Linear(int(opt.img_h/2)*int(opt.img_w/2), 16 * 16)
        self.fc_2 = nn.Linear(int(opt.img_h/4)*int(opt.img_w/4), 8 * 8)
        self.fc_3 = nn.Linear(int(opt.img_h/8)*(int(opt.img_w/4)+1), 4 * 4) # when opt.width is set to 256, width of feature_map_list[3] become 65,not 64
        self.Deconv_1 = nn.ConvTranspose2d(opt.fmap_dim + 128, 128, kernel_size = [2,2], stride = [2,2])
        self.Deconv_2 = nn.ConvTranspose2d(128, 64, kernel_size = [2,2], stride = [2,2])
        self.Deconv_3 = nn.ConvTranspose2d((64+512), 32, kernel_size = [2,2], stride= [2,2]) # ref : fmap list idx 3 (channels 512)
        self.Deconv_4 = nn.ConvTranspose2d((32+256), 16, kernel_size = [2,2], stride = [2,2])  # ref : fmap list idx 2 (channels 256)
        self.Deconv_5 = nn.ConvTranspose2d((16+128), 1, kernel_size = [2,2], stride=[2,2])  # ref : fmap list idx1 (channels 128)
        self.num_fonts = opt.num_fonts
        self.font_embedding = nn.Embedding(self.num_fonts, 128)
        self.ref_glyphs = torch.Tensor(np.load('./EFIFSTR/data/glyphs_ko_104.npy')).to(device)
        self.criterion = torch.nn.L1Loss(reduction='none').to(device)
        self.device = device
        
        
    def generate_glimpse(self, fmap, masks):
        _, fmap_c, fmap_h, fmap_w = fmap.shape
        mask = F.interpolate(masks, size = (fmap_h, fmap_w), mode ='bilinear', align_corners=False)
        mask = mask.repeat(1, fmap_c, 1, 1)
        fmap = fmap.unsqueeze(1)
        fmap = fmap.repeat(1, self.seq_len, 1, 1, 1).reshape((self.batch_size * self.seq_len, fmap_c, fmap_h, fmap_w))
        glimpse = torch.mul(mask, fmap)
        
        glimpse = glimpse.reshape((self.batch_size * self.seq_len, fmap_c, fmap_h*fmap_w))
        
        return glimpse, fmap_c
    
        
    def forward(self, feature_map_list, masks, glimpses):
        
        self.batch_size, self.seq_len, self.height, self.width = masks.shape
        masks = masks.reshape((self.batch_size * self.seq_len, 1, self.height, self.width))
        
        glimpse_s1, fmap_s1_c = self.generate_glimpse(feature_map_list[1], masks)
        glimpse_s2, fmap_s2_c = self.generate_glimpse(feature_map_list[2], masks)
        glimpse_s3, fmap_s3_c = self.generate_glimpse(feature_map_list[3], masks)
        
#         print(f'glimpse s1 shape : {glimpse_s1.shape}, s1 channels : {fmap_s1_c}')
#         print(f'glimpse s2 shape : {glimpse_s2.shape}, s1 channels : {fmap_s2_c}')
#         print(f'glimpse s3 shape : {glimpse_s3.shape}, s1 channels : {fmap_s3_c}')
        
        fmap_last = feature_map_list[-1]
        _, feature_c, feature_h, feature_w = fmap_last.shape
        
        glimpse_1 = self.fc_1(glimpse_s1).reshape((self.batch_size * self.seq_len, fmap_s1_c, 16, 16))
        glimpse_2 = self.fc_2(glimpse_s2).reshape((self.batch_size * self.seq_len, fmap_s2_c, 8,8))
        glimpse_3 = self.fc_3(glimpse_s3).reshape((self.batch_size * self.seq_len, fmap_s3_c, 4,4))
        
        embedding_ids = torch.randint(low=0, high=self.num_fonts, size=(self.batch_size * self.seq_len, 1)).to(self.device)
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
        
        glyph = d5.reshape((self.batch_size, self.seq_len, 32*32))
        
        return glyph, embedding_ids
    
    
    def glyph_loss(self, glyphs, target, target_length, embedding_ids, opt):
        target_glyph_ids = target.reshape((opt.batch_size * (opt.batch_max_length+1), 1)) + 2447 * embedding_ids
        ref_target = self.ref_glyphs[target_glyph_ids.reshape(-1,)]
        ref_target = ref_target.reshape((opt.batch_size, opt.batch_max_length+1, 32*32))
#         print(f'glyphs shape : {glyphs.shape}')
#         print(f'ref_target shape : {ref_target.shape}')
        l1_loss_ = self.criterion(glyphs, ref_target)
        l1_avg = torch.mean(l1_loss_, dim=2) # [N, max_seq +1]
        
        max_lengths = torch.Tensor(list(range(opt.batch_max_length+1))).repeat((opt.batch_size,1)).to(self.device)
        loss_mask = torch.lt(max_lengths, target_length.reshape(-1,1))
        masked_loss = torch.mul(l1_avg, loss_mask)
        row_losses = torch.sum(masked_loss, dim=1)
        loss = torch.div(torch.sum(row_losses), opt.batch_size)
        generation_loss = loss * 0.5
        
        return generation_loss
    
        
        
        
        
        
    