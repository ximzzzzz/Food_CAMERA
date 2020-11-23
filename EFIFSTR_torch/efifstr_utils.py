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

class LabelConverter(object):
    
    def __init__(self, character, device):
        list_token = ['[GO]', '[s]']
        list_character = list(character)
        self.character = list_token + list_character
        self.n_cls = len(self.character)
        self.device = device
        
        self.dict = {}
        for i, char in enumerate(self.character):
            self.dict[char] = i
    
    def encode(self, text, batch_max_length = 25):
        length = [len(s) + 1 for s in text] # +1 for [s]
        batch_max_length +=1
        batch_text = torch.LongTensor(len(text), batch_max_length ).fill_(0)
        for i, t in enumerate(text):
            text = list(t)
            text.append('[s]')
            text = [self.dict[char] for char in text]
            batch_text[i][ : len(text)] = torch.LongTensor(text)
        return (batch_text.to(self.device ), torch.IntTensor(length).to(self.device))
#         return (batch_text, torch.IntTensor(length))
    
    
    def decode(self, text_index, length):
        texts = []
        for index, i in enumerate(length):
            text = ''.join([self.character[i] for i in text_index[index, :]])
            texts.append(text)
            
        return texts