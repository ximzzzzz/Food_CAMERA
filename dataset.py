
import numpy as np
import json
import pandas as pd
from tqdm import tqdm
import os
import torchvision.transforms as transforms
from PIL import Image
import random
from torch.utils.data import *
import utils
import easydict


class CUTE_dataset(Dataset):
    def __init__(self, opt, transformer=None, ):
        self.base_path = '../../EnglishSTR/cute'
        self.gt_path = os.path.join(self.base_path, 'gt.txt')
        self.dataset = self._read_gt()
        self.transformer= transformer
        self.align_collater = self.align_collate(opt)
        
    def _read_gt(self):
        with open(self.gt_path, 'r') as f:
            gt_file = f.readlines()
            file_names = []
            labels = []
            for line in gt_file:
                file_name, label = line.strip('\n').split(' ', maxsplit=1)
                file_names.append(file_name)
                labels.append(label)
                
            images_labels = []
            for file_name,label in zip(file_names, labels):
                image = Image.open(os.path.join(self.base_path, file_name ))
                images_labels.append([image, label])
#             dataset = np.concatenate((np.asarray(images).reshape(-1,1), np.asarray(labels).reshape(-1,1)), axis=1)
            return images_labels
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        
        if self.transformer:
            return (self.transformer(self.dataset[idx][0]), self.dataset[idx][1])
        
        else:
            return self.dataset[idx]
        
    def align_collate(self, opt):
        align_colater = utils.AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=True)
        return align_colater
    
    
class IC03_dataset(Dataset):
    def __init__(self, opt, transformer=None):
        self.base_path = '../../EnglishSTR/ic03'
        self.gt_path = os.path.join(self.base_path, 'gt_lex50.txt')
        self.dataset = self._read_gt()
        
    def _read_gt(self):
        with open(self.gt_path, 'r') as f:
            gt_file = f.readlines()
            file_names = []
            labels = []
            for line in gt_file:
                splited = line.split()
                file_names.append(splited[0])
                labels.append(splited[1])
            images_labels = []
            for file_name, label in zip(file_names, labels):
                image = Image.open(os.path.join(self.base_path, file_name))
                images_labels.append([image, label])
            return images_labels
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        
        if self.transformer:
            return (self.transformer(self.dataset[idx][0]), self.dataset[idx][1])
        
        else:
            return self.dataset[idx]
        
    def align_collate(self, opt):
        align_colater = utils.AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=True)
        return align_colater