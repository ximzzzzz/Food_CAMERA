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


class Attention_unit(nn.Module):
    
    def __init__(self, fmap_dim, lstm_dim, attn_dim):
        super(Attention_unit, self).__init__()
        self.fmap_dim = fmap_dim
        self.lstm_dim = lstm_dim
        self.e_lstm_conv = nn.Conv2d(lstm_dim, attn_dim, kernel_size=1)
        self.e_Fmap_conv = nn.Conv2d(fmap_dim, attn_dim, kernel_size=3, padding=1)
        self.a_conv = nn.Conv2d(attn_dim, 1, kernel_size=1)
        
    def forward(self, fmap, hidden_state):
        batch_size, channel, height, width = fmap.shape
        hidden_state = hidden_state.permute(0, 2, 1).unsqueeze(3)
        e_lstm_conv_ =  self.e_lstm_conv(hidden_state)
        e_lstm_conv_ = e_lstm_conv_.repeat(1, 1, height, width)
        e_fmap_conv_ = self.e_Fmap_conv(fmap)
#         print('e_lstm_conv res : ', e_lstm_conv_.shape)
#         print('e_fmap_conv res : ', e_fmap_conv_.shape)
        e = torch.tanh_(e_lstm_conv_ + e_fmap_conv_)
        a_conv_ = self.a_conv(e)  
        a = F.softmax(a_conv_.reshape((batch_size, -1)), dim = -1)
        mask = a.reshape((batch_size, 1, height, width))
        broad_casted = (fmap * mask).reshape(batch_size, channel, -1)
        glimpse = torch.sum(broad_casted, dim= -1).reshape((batch_size, channel, 1, 1))
#         print('glimpse shape : ', glimpse.shape)
        return glimpse, mask


class Decoder(nn.Module): 
    
    def __init__(self, opt, device): #att dim = 512
        super(Decoder, self).__init__()
        self.num_classes = opt.num_classes
        self.fmap_dim = opt.fmap_dim
        self.enc_dim = opt.enc_dim
        self.dec_dim = opt.dec_dim
        self.attn_dim = opt.attn_dim
        self.max_length = opt.max_length
        self.attention_unit = Attention_unit(self.fmap_dim, self.dec_dim, self.attn_dim )
        self.input_embedding = nn.Embedding(self.num_classes+1, self.enc_dim) # including <BOS>
#         self.lstm = nn.LSTMCell(enc_dim, dec_dim)
        self.lstm = nn.LSTM(self.enc_dim, self.dec_dim, num_layers = 2, batch_first=True)
        self.fc = nn.Linear(self.dec_dim + self.fmap_dim, self.num_classes +2 ) # including <BOS>,<EOS>
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=0).to(device)
        self.device = device
        
        
    def forward(self, feature_map, holistic_feature, Input, is_train):
#         x, target, length = Input
        target, length = Input
        batch_size, channel, height, width = feature_map.shape

        logits = torch.zeros(batch_size, self.max_length+1, self.num_classes+2).to(self.device )  ### including BOS, EOS
        masks = torch.zeros(batch_size, self.max_length+1, height, width ).to(self.device )
        glimpses = torch.zeros(batch_size, self.max_length+1, channel).to(self.device )
        input_label = torch.zeros(batch_size, 1, dtype= torch.long).to(self.device ) # zero means BOS

        input_emb = self.input_embedding(input_label).to(self.device )
        self.lstm.flatten_parameters()
        output, states =  self.lstm(input_emb, holistic_feature)
        glimpse, mask = self.attention_unit(feature_map, output)
        glimpse = glimpse.permute(0, 2, 1, 3).squeeze(3)
        logit = self.fc(torch.cat([output, glimpse], axis=2))
        logits[ :, [0], :] = logit
        masks[  :, [0], :, : ] = mask
        glimpses[:,[0], : ] = glimpse
        
        if is_train:
            for i in range(self.max_length):
                input_label = target[:, [i]]
                input_emb = self.input_embedding(input_label)
                self.lstm.flatten_parameters()
                output, states = self.lstm(input_emb, states)
                glimpse, mask = self.attention_unit(feature_map, output)
                glimpse = glimpse.permute(0, 2, 1, 3).squeeze(3)
                logit = self.fc(torch.cat([output, glimpse], axis=2))
                
                logits[:, [i+1], :] = logit
                masks[:,[i+1], :, : ] = mask
                glimpses[:, [i+1], : ] = glimpse
        else:
            pred = torch.argmax(logit, dim=-1)
            for i in range(1, self.max_length):
                input_emb = self.input_embedding(pred)
                self.lstm.flatten_parameters()
                output, states = self.lstm(input_emb, states)
                glimpse, mask = self.attention_unit(feature_map, output)
                glimpse = glimpse.permute(0, 2, 1, 3).squeeze(3)
                logit = self.fc(torch.cat([output, glimpse], axis=2))
                pred = torch.argmax(torch.softmax(logit, axis=-1), -1)
                
                logits[:, [i+1], :] = logit 
                masks[:,[i+1], :, : ] = mask
                glimpses[:, [i+1], : ] = glimpse
                
        return logits, masks, glimpses
    
    def recognition_loss(self, logits, target):
        recognition_loss_ = self.criterion(logits, target)
        
        return recognition_loss_
        