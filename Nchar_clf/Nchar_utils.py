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
from albumentations.core.transforms_interface import DualTransform
from albumentations.augmentations import functional as F
sys.path.append('/home/Data/FoodDetection/AI_OCR/Whatiswrong')
sys.path.append('/home/Data/FoodDetection/AI_OCR/Scatter')
from utils import *
# from augs import *

class CustomDataset_clf(Dataset):
    
    def __init__(self, dataset,device, resize_shape = (64, 256), input_channel = 3, is_module=False, is_train = True ):
        self.dataset = dataset
        self.resize_H = resize_shape[0]
        self.resize_W = resize_shape[1]
        self.resize_max_W = int(self.resize_H/4) * 23
        self.totensor = transforms.ToTensor()
        self.is_train = is_train
        self.is_module = is_module  #### for robust scanner dataset
        
        # use for training additional model or training itself
        if self.is_module:
#             name = 'efficientnet-b3'
            name = 'efficientnet-b0'
            self.model = EfficientNet.from_name(name, include_top=True)
            self.model._fc = torch.nn.Linear(in_features = self.model._fc.in_features, out_features = 24, bias=True)
#             previous_iter = get_latest_model(name, './Nchar_clf/models')
#             load_path = f'./Nchar_clf/models/{name}_{previous_iter}.pth'
#             load_path = f'./models/Nchar_clf_1108/0/bestacc_0.96_0.pth'
            load_path = './models/Nchar_clf_1107/0/bestacc_0.957_14000.pth'
            print(f'{load_path} is loaded for n_char model')
            if load_path :
                self.model.load_state_dict(torch.load(load_path, map_location='cpu'))
#             self.model = torch.nn.DataParallel(self.model, device_ids = [0,1]).to(device)
            self.model = self.model.to(torch.device('cpu'))
            self.model = self.model.eval()
            
        
        # for augmentation setting
        if self.is_train:
            self.transform = albumentations.Compose([
                    GridMask(num_grid = 10, p=0.7),
                albumentations.RandomBrightnessContrast(p=0.5),
#                 albumentations.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.8, p=0.5 ),
                albumentations.augmentations.transforms.Rotate(limit=10, p=0.5),
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
            skip_flag, rotate_flag = self.weird_img_scanner(image)
            if skip_flag:
                img_path, top, mid, bot = self.dataset[idx-1]
                image = self.load_and_cvt(img_path)
                rotate_flag=False
                
            if rotate_flag  :
                image = cv2.rotate(image,  cv2.ROTATE_90_CLOCKWISE)
                
            image_trsf= self.transform(image=image)['image']
            image_tensor = torch.Tensor(image_trsf).unsqueeze_(0)
            
            pred_nchar = self.model(image_tensor)
            pred_cls = torch.argmax(torch.softmax(pred_nchar, -1), -1)
            nchar = pred_cls.cpu().detach().numpy()[0]
#             print('nchar pred : ', nchar)
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
            skip_flag, rotate_flag = self.weird_img_scanner(image)
            if skip_flag:
                img_path, label = self.dataset[idx-1]
                image = self.load_and_cvt(img_path)
                rotate_flag=False
            if rotate_flag :
                image = cv2.rotate(image,  cv2.ROTATE_90_CLOCKWISE)
                
            image = self.transform(image=image)['image']

            return image, len(label)
    
    def load_and_cvt(self, img_path):
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    
    def weird_img_scanner(self, image):
        skip_flag= False
        rotate_flag = False
        height, width, _ = image.shape
        if (height==1) | (width==1):
            skip_flag = True
            
        if height > width:
            rotate_flag = True
        return skip_flag, rotate_flag

    
    
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

class GridMask(DualTransform):

    def __init__(self, num_grid , fill_value=0, rotate=0, mode=0, always_apply=False, p=0.5):
        super(GridMask, self).__init__(always_apply, p)
        if isinstance(num_grid, int):
            num_grid = (num_grid, num_grid)
        if isinstance(rotate, int):
            rotate = (-rotate, rotate)
        self.num_grid = num_grid
        self.fill_value = fill_value
        self.rotate = rotate
        self.mode = mode
        self.masks = None
        self.rand_h_max = []
        self.rand_w_max = []

    def init_masks(self, height, width):
#         if self.masks is None:
        self.masks = []
#         n_masks = self.num_grid[1] - self.num_grid[0] + 1
        ratio = width/height if width/height > 1 else 1
        for n, n_g in enumerate(range(self.num_grid[0], self.num_grid[1] + 1, 1)):
  
            n_g_width = int(ratio * n_g) 
            grid_h = height / n_g if height / n_g >0 else 1
#             grid_w = width / n_g
            grid_w = width / n_g_width if width / n_g_width >0 else 1
#             print(f'grid_h : {grid_h}, grid_w : {grid_w}')
            this_mask = np.ones((int((n_g + 20) * grid_h), int((n_g_width + 20) * grid_w))).astype(np.uint8)
#             print('mask size : ', this_mask.shape)
            for i in range(n_g + 1):
                for j in range(n_g_width + 1):
                    this_mask[
                         int(i * grid_h) : int(i * grid_h + grid_h / 2),
                         int(j * grid_w) : int(j * grid_w + grid_w / 2)
                    ] = self.fill_value
                    if self.mode == 2:
                        this_mask[
                             int(i * grid_h + grid_h / 2) : int(i * grid_h + grid_h),
                             int(j * grid_w + grid_w / 2) : int(j * grid_w + grid_w)
                        ] = self.fill_value
            if self.mode == 1:
                this_mask = 1 - this_mask

            self.masks.append(this_mask)
            self.rand_h_max.append(grid_h)
            self.rand_w_max.append(grid_w)

    def apply(self, image, mask, rand_h, rand_w, angle, **params):
        h, w = image.shape[:2]
#         print(f'h : {h}, w :{w}')
        mask = F.rotate(mask, angle) if self.rotate[1] > 0 else mask
        mask = mask[:,:,np.newaxis] if image.ndim == 3 else mask
#         print('mask shape : ', mask[rand_h:rand_h+h, rand_w:rand_w+w].astype(image.dtype).shape)
#         print('img shape : ', image.shape)
        try:
            image *= mask[rand_h:rand_h+h, rand_w:rand_w+w].astype(image.dtype)
        except :
            pass
        return image

    def get_params_dependent_on_targets(self, params):
        img = params['image']
        height, width = img.shape[:2]
#         init_n_g_height = 10
#         ratio = width/height
#         init_n_g_width = int(ratio * init_n_g_height)
#         self.num_grid = (init_n_g_height, init_n_g_width)
        self.init_masks(height, width)

        mid = np.random.randint(len(self.masks))
        mask = self.masks[mid]
        rand_h = np.random.randint(int(self.rand_h_max[mid]))
        rand_w = np.random.randint(int(self.rand_w_max[mid]))
        angle = np.random.randint(self.rotate[0], self.rotate[1]) if self.rotate[1] > 0 else 0
#         print(f'rand_h : {rand_h}, rand_w : {rand_w}')

        return {'mask': mask, 'rand_h': rand_h, 'rand_w': rand_w, 'angle': angle}

    @property
    def targets_as_params(self):
        return ['image']

    def get_transform_init_args_names(self):
        return ('num_grid', 'fill_value', 'rotate', 'mode')