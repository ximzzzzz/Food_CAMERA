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
import Trans
import Nchar_utils
import Extract
import utils
import evaluate
import torch.nn.functional as F
from torch.utils.data import *
import easydict
import torchvision
import tensorflow as tf
import pickle
import time
import os
import Decoder
import Encoder
import GlyphGen

class Basemodel(nn.Module):
    def __init__(self, opt, device):
        super(Basemodel, self).__init__()
        
        if opt.TPS:
            self.TPS = Trans.TPS_SpatialTransformerNetwork(F = opt.num_fiducial,
                                                      i_size = (opt.img_h, opt.img_w), 
                                                      i_r_size= (opt.img_h, opt.img_w), 
                                                      i_channel_num= 3, #input channel 
                                                            device = device)
        self.encoder = Encoder.Resnet_encoder(opt)
        self.decoder = Decoder.Decoder(opt,device)
        self.generator = GlyphGen.Generator(opt, device)
        
        
    def forward(self, img, Input,is_train):
        
        if self.TPS:
            img = self.TPS(img)
            
        feature_map_list, holistic_states = self.encoder(img)
        logits, masks, glimpses = self.decoder(feature_map_list[-1], holistic_states, Input, is_train)
        glyphs, embedding_ids = self.generator(feature_map_list, masks, glimpses)
        
        return logits, glyphs, embedding_ids