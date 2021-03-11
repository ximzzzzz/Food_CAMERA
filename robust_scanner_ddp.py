#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
sys.path.append('./Whatiswrong')
sys.path.append('./Scatter')
sys.path.append('./RobustScanner')

import re
import six
import math
import torchvision.transforms as transforms

import utils
from utils import *
import augs
import augs2
import BaseModel
import torch.distributed as dist
import en_dataset
import ko_dataset
from albumentations import GaussNoise, IAAAdditiveGaussianNoise, Compose, OneOf
from albumentations.pytorch import ToTensor
import albumentations
import evaluate
import cv2


# In[2]:


# opt
opt = easydict.EasyDict({
    "experiment_name" : f'{utils.SaveDir_maker(base_model = "RobustScanner", base_model_dir = "./models")}',
    'saved_model' : 'RobustScanner_0309/3/best_accuracy_97.33.pth',
#     'saved_model' : '',
    "manualSeed" : 1111,
    "imgH" : 64 , "imgW" :  256,
    "PAD" : True ,
    'batch_size' : 384,
    'data_filtering_off' : True,
    'workers' : 20,
    'rgb' :True,
    'sensitive' : True,
    'top_char' : ' !"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZㄱㄴㄷㄹㅁㅂㅅㅇㅈㅊㅋㅌㅍㅎㄲㄸㅃㅆㅉ',
    'middle_char' : ' ㅏㅑㅓㅕㅗㅛㅜㅠㅡㅣㅐㅒㅔㅖㅘㅙㅚㅝㅞㅟㅢ',
    'bottom_char' : ' ㄱㄴㄷㄹㅁㅂㅅㅇㅈㅊㅋㅌㅍㅎㄲㄸㅃㅆㅉㄳㄵㄶㄺㄻㄼㄽㄾㄿㅀㅄ',
    'batch_max_length' : 23, ############ 25-> 23
    'num_fiducial' : 20,
    'output_channel' : 512,
    'hidden_size' :256,
#     'optimizer' : 'adam',
    'lr' : 1,
    'rho' : 0.95,
    'eps' : 1e-8,
    'grad_clip' : 5,
    'valInterval' : 1500,
    'num_epoch' : 5,
    'input_channel' : 3,
    'FT' : True,
    'extract' : 'resnet', 'pred' : ' ', 'trans' : True, 'hybrid_direction' : 2, 'position_direction' : 2
    })
# device = torch.device('cuda') #utils.py 안에 device는 따로 세팅해줘야함
device = torch.device('cpu') #utils.py 안에 device는 따로 세팅해줘야함
top_converter = utils.AttnLabelConverter(opt.top_char, device)
middle_converter = utils.AttnLabelConverter(opt.middle_char, device)
bottom_converter = utils.AttnLabelConverter(opt.bottom_char, device)
opt.top_n_cls = len(top_converter.character)
opt.mid_n_cls = len(middle_converter.character)
opt.bot_n_cls = len(bottom_converter.character)


# In[3]:


def get_train_loader(opt, num_worker):
    with open('./dataset_ADDhard4', 'rb') as file:  # 20/02/08 추가 데이터셋으로 도전 
        data = pickle.load(file)
    
    transformers = Compose([
                        OneOf([
                                  augs.VinylShining(1),
                            augs.GridMask(num_grid=(15,15)),
                            augs.RandomAugMix(severity=2, width=2),
#                             augs.LensDistortion(0.8, 1.2),
#                             augs.BarrelDistortion()
                            ], p =1.0), # 2021-02-08 전 버전은 severity, width 1 로 했었음
                            ToTensor()
                       ])
    train_custom = utils.CustomDataset_jamo(data[ : int(len(data) * 0.99)], resize_shape = (opt.imgH, opt.imgW), transformer=transformers)
    test_custom = utils.CustomDataset_jamo(data[ int(len(data) * 0.99): ], resize_shape = (opt.imgH, opt.imgW), transformer=ToTensor())
    
    train_sampler = torch.utils.data.DistributedSampler(train_custom)
    test_sampler = torch.utils.data.DistributedSampler(test_custom)
    
    train_loader = DataLoader(train_sampler, opt.batch_size, pin_memory=True, num_workers=num_worker, shuffle=False, sampler=train_sampler)
    test_loader = DataLoader(test_sampler, opt.batch_size, pin_memory=True, num_workers=num_worker, shuffle=False, sampler=test_sampler)
    
    return train_loader, test_loader


# In[4]:


def main():
    ngpus_per_node = torch.cuda.device_count()
    world_size = ngpus_per_node
    
    torch.multiprocessing.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, opt))


# In[11]:


def main_worker(gpu, *args):
    ngpus_per_node, opt = args
    
    num_worker = 20
    batch_size = int(opt.batch_size / ngpus_per_node)
    num_worker = int(num_worker / ngpus_per_node)
    
#     os.environ['MASTER_ADDR'] = '127.0.0.1'
#     os.environ['MASTER_PORT'] = '2222'
    
    torch.distributed.init_process_group(
        backend='nccl',
#         init_method = 'tcp://127.0.0.1:2222',
        init_method='env://',
        world_size = ngpus_per_node,
        rank = gpu)
    
    model = BaseModel.model(opt, device)
    
    # load pretrained model
    if opt.saved_model != '':
        base_path = './models'
        print(f'looking for pretrained model from {os.path.join(base_path, opt.saved_model)}')
        try :
            model.load_state_dict(torch.load(os.path.join(base_path, opt.saved_model), map_location = 'cpu' if device.type=='cpu' else None))
            print('loading complete ')    
        except Exception as e:
            print(e)
            print('coud not load model')
            
    model = model.cuda(gpu)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids = [gpu])
    
    train_loader, test_loader = get_train_loader(opt, num_worker)
    
    # loss
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0).to(device) #ignore [GO] token = ignore index 0
    log_avg = utils.Averager()
    
    # filter that only require gradient descent
    filtered_parameters = []
    params_num = []
    for p in filter(lambda p : p.requires_grad, model.parameters()):
        filtered_parameters.append(p)
        params_num.append(np.prod(p.size()))
    print('Tranable params : ', sum(params_num))
    
    # optimizer
    optimizer = optim.Adadelta(filtered_parameters, lr= opt.lr, rho = opt.rho, eps = opt.eps)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', verbose=True, patience = 2, factor= 0.5 )
    
    #start training
    start_time = time.time()
    best_accuracy = -1
    best_norm_ED = -1
    swa_count = 0
    
    for n_epoch, epoch in enumerate(range(opt.num_epoch)):
        for n_iter, data_point in enumerate(train_loader):
            
            image_tensors, top, mid, bot = data_point 

            image = image_tensors.to(gpu)
            text_top, length_top = top_converter.encode(top, batch_max_length = opt.batch_max_length)
            text_mid, length_mid = middle_converter.encode(mid, batch_max_length = opt.batch_max_length)
            text_bot, length_bot = bottom_converter.encode(bot, batch_max_length = opt.batch_max_length)
            
            text_top, length_top = text_top.to(gpu), length_top.to(gpu)
            text_mid, length_mid = text_mid.to(gpu), length_mid.to(gpu)
            text_bot, length_bot = text_bot.to(gpu), length_bot.to(gpu)
            
            batch_size = image.size(0)
          
            pred_top, pred_mid, pred_bot = model(image, text_top[:,:-1], text_mid[:,:-1], text_bot[:,:-1])
            
#             cost_top = criterion(pred_top.view(-1, pred_top.shape[-1]), text_top[:, 1:].contiguous().view(-1))
#             cost_mid = criterion(pred_mid.view(-1, pred_mid.shape[-1]), text_mid[:, 1:].contiguous().view(-1))
#             cost_bot = criterion(pred_bot.view(-1, pred_bot.shape[-1]), text_bot[:, 1:].contiguous().view(-1))
        
            cost_top = utils.reduced_focal_loss(pred_top.view(-1, pred_top.shape[-1]), text_top[:, 1:].contiguous().view(-1), gamma=2,  threshold=0.5)
            cost_mid = utils.reduced_focal_loss(pred_mid.view(-1, pred_mid.shape[-1]), text_mid[:, 1:].contiguous().view(-1), gamma=2,  threshold=0.5)
            cost_bot = utils.reduced_focal_loss(pred_bot.view(-1, pred_bot.shape[-1]), text_bot[:, 1:].contiguous().view(-1), gamma=2,  threshold=0.5)
            
#             cost_top = utils.CB_loss(text_top[:, 1:].contiguous().view(-1), pred_top.view(-1, pred_top.shape[-1]), top_per_cls, opt.top_n_cls, 'focal', 0.99, 2)
#             cost_mid = utils.CB_loss(text_mid[:, 1:].contiguous().view(-1), pred_mid.view(-1, pred_mid.shape[-1]), mid_per_cls, opt.mid_n_cls, 'focal', 0.99, 2)
#             cost_bot = utils.CB_loss(text_bot[:, 1:].contiguous().view(-1), pred_bot.view(-1, pred_bot.shape[-1]), bot_per_cls, opt.bot_n_cls, 'focal', 0.99, 2)
            cost = cost_top + cost_mid + cost_bot

#             print('Cost top : ', cost_top)
#             print('Cost mid : ', cost_mid)
#             print('Cost bot : ', cost_bot)
            loss_avg = utils.Averager()
            loss_avg.add(cost)
            
            model.zero_grad()
            cost.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip) #gradient clipping with 5
            optimizer.step()
            
            print(f'epoch : {epoch} | step : {n_iter} / {len(train_loader)} | mp : {gpu}')
            
            


# In[12]:


main()


# In[ ]:




