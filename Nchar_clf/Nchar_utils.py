import os
import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.optim as optim
import torch.utils.data
from torch.utils.data import *
import numpy as np
import time
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import easydict
import sys
import pickle
import re
import six
import math
import torchvision.transforms as transforms
import torch.distributed as dist
from albumentations import GaussNoise, IAAAdditiveGaussianNoise, Compose, OneOf
from albumentations.pytorch import ToTensor
import albumentations
import cv2
from torch.utils.data import Dataset, ConcatDataset, Subset
import os
from efficientnet_pytorch import EfficientNet

class CustomDataset_clf(Dataset):
    
    def __init__(self, dataset,device, resize_shape = (64, 256), input_channel = 3, is_module=False, is_train = True ):
        self.dataset = dataset
        self.resize_H = resize_shape[0]
        self.resize_W = resize_shape[1]
        self.resize_max_W = int(self.resize_H/4) * 23
        self.totensor = transforms.ToTensor()
        self.is_train = is_train
        self.is_module = is_module  #### for robust scanner dataset
        
        if self.is_module:
            name = 'efficientnet-b0'
            self.model = EfficientNet.from_name(name, include_top=True)
            self.model._fc = torch.nn.Linear(in_features = 1280, out_features = 23, bias=True)
            previous_iter = get_latest_model(name, './Nchar_clf/models')
            load_path = f'./Nchar_clf/models/{name}_{previous_iter}.pth'
            
            if load_path :
                self.model.load_state_dict(torch.load(load_path))
            self.model = torch.nn.DataParallel(self.model, device_ids = [0,1]).to(device)
            self.model = self.model.eval()
            
        if self.is_train:
            self.transform = albumentations.Compose([
                albumentations.RandomBrightnessContrast(p=0.5),
    #             albumentations.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.8, p=0.5 ),
                albumentations.Resize(self.resize_H, self.resize_W),
                albumentations.Normalize( mean=[0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),
                albumentations.pytorch.transforms.ToTensor()
            ])
        else:
            self.transform = albumentations.Compose([
                albumentations.Resize(self.resize_H, self.resize_W),
                albumentations.Normalize( mean=[0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),
                albumentations.pytorch.transforms.ToTensor()
            ])
        

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        if self.is_module:
            img_path, top, mid, bot = self.dataset[idx]
            image = self.load_and_cvt(img_path)
            image_trsf= self.transform(image=image)['image']
            image_tensor = torch.Tensor(image_trsf).unsqueeze_(0)
            
            pred_nchar = self.model(image_tensor)
            pred_cls = torch.argmax(torch.softmax(pred_nchar, -1), -1)
            nchar = pred_cls.cpu().detach().numpy()[0]
            resize_char_width = int(self.resize_H/4) * nchar
            image_resize = cv2.resize(image, (resize_char_width, self.resize_H))
            image_tensor = self.totensor(image_resize)
            c, h , w = image_tensor.size()
            pad_img = torch.FloatTensor(*(3, self.resize_H, self.resize_max_W)).fill_(0)
            pad_img[:, :, :w] =image_tensor
            
            if self.resize_max_W != w:
                pad_img[:,:,w:] = image_tensor[:,:,-1].unsqueeze(2).expand(c,h, self.resize_max_W - w)
                
            top = make_str(top)
            mid = make_str(mid)
            bot = make_str(bot)
            
            return (pad_img, top, mid, bot)

            
        else:
            img_path, label = self.dataset[idx]
            image = self.load_and_cvt(img_path)
            image = self.transform(image=image)['image']

            return image, len(label)
    
    def load_and_cvt(self, img_path):
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    
    
def get_accuracy(pred, label):
    pred_max = torch.argmax(torch.softmax(pred, -1), -1)
    match = 0
    batch_size = pred_max.shape[0]
    for i in range(batch_size):
        if pred_max[i] == label[i]:
            match +=1
    return round(match / batch_size, 3)

def get_latest_model(name, model_directory):
    relate_models = []
    for model_file in os.listdir(model_directory):
        if re.compile(name).match(model_file):
            relate_models.append(int(model_file.split('_')[-1].replace('.pth', '')))
    return max(relate_models)  

def make_str(label):
    string = ''
    for lab in label:
        string+=lab
    return string