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

import re
import six
import math
import torchvision.transforms as transforms

import utils
from utils import *
import augs
import www_model_jamo_vertical
import torch.distributed as dist
import en_dataset
import ko_dataset
from albumentations import GaussNoise, IAAAdditiveGaussianNoise, Compose, OneOf
from albumentations.pytorch import ToTensor
import evaluate
from torchcontrib.optim import SWA


# In[2]:


import importlib
importlib.reload(ko_dataset)
importlib.reload(en_dataset)
importlib.reload(utils)


# In[3]:


# GPU_NUM = 1 
# device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
# torch.cuda.set_device(device) # change allocation of current GPU
# print ('Current cuda device ', torch.cuda.current_device()) # check


# ### arguements

# In[2]:


# opt
opt = easydict.EasyDict({
    "experiment_name" : f'{utils.SaveDir_maker(base_model = "www_jamo_vertical", base_model_dir = "./models")}',
    'saved_model' : '',
    "manualSeed" : 1111,
    "imgH" : 35 ,
    "imgW" :  250,
    "PAD" : True ,
    'batch_size' : 192,
    'data_filtering_off' : True,
    'workers' : 20,
    'rgb' :True,
    'sensitive' : True,
    'top_char' : ' !"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZㄱㄴㄷㄹㅁㅂㅅㅇㅈㅊㅋㅌㅍㅎㄲㄸㅃㅆㅉ',
    'middle_char' : ' ㅏㅑㅓㅕㅗㅛㅜㅠㅡㅣㅐㅒㅔㅖㅘㅙㅚㅝㅞㅟㅢ',
    'bottom_char' : ' ㄱㄴㄷㄹㅁㅂㅅㅇㅈㅊㅋㅌㅍㅎㄲㄸㅃㅆㅉㄳㄵㄶㄺㄻㄼㄽㄾㄿㅀㅄ',
    'batch_max_length' : 25,
    'num_fiducial' : 20,
    'output_channel' : 512,
    'hidden_size' :256,
#     'optimizer' : 'adam',
    'lr' : 1,
    'rho' : 0.95,
    'eps' : 1e-8,
    'grad_clip' : 5,
    'valInterval' : 5000,
    'num_epoch' : 100,
    'input_channel' : 3,
    'FT' : True,
    # 'extract' : 'RCNN',
    'extract' : 'efficientnet-b5',
    'pred' : ' '
    })

top_converter = utils.AttnLabelConverter(opt.top_char)
middle_converter = utils.AttnLabelConverter(opt.middle_char)
bottom_converter = utils.AttnLabelConverter(opt.bottom_char)
opt.top_n_cls = len(top_converter.character)
opt.middle_n_cls = len(middle_converter.character)
opt.bottom_n_cls = len(bottom_converter.character)
device = torch.device('cuda') #utils.py 안에 device는 따로 세팅해줘야함


with open('./dataset', 'rb') as file:
    data = pickle.load(file)


# In[4]:


transformers = Compose([
                        OneOf([
#                                   augs.VinylShining(1),
                            augs.GridMask(num_grid=(10,20)),
                            augs.RandomAugMix(severity=2, width=2)], p =0.7),
                            ToTensor()
                       ])
train_custom = utils.CustomDataset_jamo(data[ : int(len(data) * 0.98)], resize_shape = (opt.imgH, opt.imgW), transformer=transformers)
valid_custom = utils.CustomDataset_jamo(data[ int(len(data) * 0.98): ], resize_shape = (opt.imgH, opt.imgW), transformer=ToTensor())

data_loader = DataLoader(train_custom, batch_size = opt.batch_size,  num_workers =15, shuffle=True, drop_last=True, pin_memory=True)
valid_loader = DataLoader(valid_custom, batch_size = opt.batch_size,  num_workers=10, shuffle=True,  drop_last=True, pin_memory=True )


# ## train


with open('top_per_cls', 'rb') as file:
    top_per_cls = pickle.load(file)
with open('mid_per_cls', 'rb') as file:
    mid_per_cls = pickle.load(file)
with open('bot_per_cls', 'rb') as file:
    bot_per_cls = pickle.load(file)


# In[8]:


def train(opt):
    model = www_model_jamo_vertical.STR(opt, device)
    print('model parameters. height {}, width {}, num of fiducial {}, input channel {}, output channel {}, hidden size {},     batch max length {}'.format(opt.imgH, opt.imgW, opt.num_fiducial, opt.input_channel, opt.output_channel, opt.hidden_size, opt.batch_max_length))
    
    # weight initialization
    for name, param, in model.named_parameters():
        if 'localization_fc2' in name:
            print(f'Skip {name} as it is already initializaed')
            continue
        try:
            if 'bias' in name:
                init.constant_(param, 0.0)
            elif 'weight' in name:
                init.kaiming_normal_(param)
                
        except Exception as e :
            if 'weight' in name:
                param.data.fill_(1)
            continue
            
    # load pretrained model
    if opt.saved_model != '':
        base_path = './models'
        print(f'looking for pretrained model from {os.path.join(base_path, opt.saved_model)}')
        
        try :
            model.load_state_dict(torch.load(os.path.join(base_path, opt.saved_model)))
            print('loading complete ')    
        except Exception as e:
            print(e)
            print('coud not find model')
            
    #data parallel for multi GPU
    model = torch.nn.DataParallel(model).to(device)
    model.train() 
     
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
    
#     base_opt = optim.Adadelta(filtered_parameters, lr= opt.lr, rho = opt.rho, eps = opt.eps)
    base_opt = torch.optim.Adam(filtered_parameters, lr=0.001)
    optimizer = SWA(base_opt)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', verbose=True, patience = 2, factor= 0.5 )
#     optimizer = adabound.AdaBound(filtered_parameters, lr=1e-3, final_lr=0.1)
    
    # opt log
    with open(f'./models/{opt.experiment_name}/opt.txt', 'a') as opt_file:
        opt_log = '---------------------Options-----------------\n'
        args = vars(opt)
        for k, v in args.items():
            opt_log +=f'{str(k)} : {str(v)}\n'
        opt_log +='---------------------------------------------\n'
        opt_file.write(opt_log)
        
    #start training
    start_time = time.time()
    best_accuracy = -1
    best_norm_ED = -1
    swa_count = 0
    
    for n_epoch, epoch in enumerate(range(opt.num_epoch)):
        for n_iter, data_point in enumerate(data_loader):
            
            image_tensors, top, mid, bot = data_point 

            image = image_tensors.to(device)
            text_top, length_top = top_converter.encode(top, batch_max_length = opt.batch_max_length)
            text_mid, length_mid = middle_converter.encode(mid, batch_max_length = opt.batch_max_length)
            text_bot, length_bot = bottom_converter.encode(bot, batch_max_length = opt.batch_max_length)
            batch_size = image.size(0)
          
            pred_top, pred_mid, pred_bot = model(image, text_top[:,:-1], text_mid[:,:-1], text_bot[:,:-1])
            
#             cost_top = criterion(pred_top.view(-1, pred_top.shape[-1]), text_top[:, 1:].contiguous().view(-1))
#             cost_mid = criterion(pred_mid.view(-1, pred_mid.shape[-1]), text_mid[:, 1:].contiguous().view(-1))
#             cost_bot = criterion(pred_bot.view(-1, pred_bot.shape[-1]), text_bot[:, 1:].contiguous().view(-1))
            if n_iter%2==0:
        
                cost_top = utils.reduced_focal_loss(pred_top.view(-1, pred_top.shape[-1]), text_top[:, 1:].contiguous().view(-1), ignore_index=0, gamma=2, alpha=0.25, threshold=0.5)
                cost_mid = utils.reduced_focal_loss(pred_mid.view(-1, pred_mid.shape[-1]), text_mid[:, 1:].contiguous().view(-1), ignore_index=0, gamma=2, alpha=0.25, threshold=0.5)
                cost_bot = utils.reduced_focal_loss(pred_bot.view(-1, pred_bot.shape[-1]), text_bot[:, 1:].contiguous().view(-1), ignore_index=0, gamma=2, alpha=0.25, threshold=0.5)
            else:
                cost_top = utils.CB_loss(text_top[:, 1:].contiguous().view(-1), pred_top.view(-1, pred_top.shape[-1]), top_per_cls, opt.top_n_cls, 'focal', 0.999, 0.5)
                cost_mid = utils.CB_loss(text_mid[:, 1:].contiguous().view(-1), pred_mid.view(-1, pred_mid.shape[-1]), mid_per_cls, opt.middle_n_cls, 'focal', 0.999, 0.5)
                cost_bot = utils.CB_loss(text_bot[:, 1:].contiguous().view(-1), pred_bot.view(-1, pred_bot.shape[-1]), bot_per_cls, opt.bottom_n_cls, 'focal', 0.999, 0.5)
            cost = cost_top*0.33 + cost_mid *0.33 + cost_bot*0.33
    
            loss_avg = utils.Averager()
            loss_avg.add(cost)
            
            model.zero_grad()
            cost.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip) #gradient clipping with 5
            optimizer.step()
            print(loss_avg.val())

            #validation
            if (n_iter % opt.valInterval == 0) & (n_iter!=0) :
                elapsed_time = time.time() - start_time
                with open(f'./models/{opt.experiment_name}/log_train.txt', 'a') as log:
                    model.eval()
                    with torch.no_grad():
                        valid_loss, current_accuracy, current_norm_ED, pred_top_str, pred_mid_str, pred_bot_str, label_top, label_mid, label_bot, infer_time, length_of_data = evaluate.validation_jamo(model, criterion, valid_loader, top_converter, middle_converter, bottom_converter, opt)
                    scheduler.step(current_accuracy)
                    model.train()

                    present_time = time.localtime()
                    loss_log = f'[epoch : {n_epoch}/{opt.num_epoch}] [iter : {n_iter*opt.batch_size} / {int(len(data) * 0.95)}]\n'+                    f'Train loss : {loss_avg.val():0.5f}, Valid loss : {valid_loss:0.5f}, Elapsed time : {elapsed_time:0.5f}, Present time : {present_time[1]}/{present_time[2]}, {present_time[3]+9} : {present_time[4]}'
                    loss_avg.reset()

                    current_model_log = f'{"Current_accuracy":17s}: {current_accuracy:0.3f}, {"current_norm_ED":17s}: {current_norm_ED:0.2f}'

                    #keep the best
                    if current_accuracy > best_accuracy:
                        best_accuracy = current_accuracy
                        torch.save(model.module.state_dict(), f'./models/{opt.experiment_name}/best_accuracy_{round(current_accuracy,2)}.pth')

                    if current_norm_ED > best_norm_ED:
                        best_norm_ED = current_norm_ED
                        torch.save(model.module.state_dict(), f'./models/{opt.experiment_name}/best_norm_ED.pth')

                    best_model_log = f'{"Best accuracy":17s}: {best_accuracy:0.3f}, {"Best_norm_ED":17s}: {best_norm_ED:0.2f}'
                    loss_model_log = f'{loss_log}\n{current_model_log}\n{best_model_log}'
                    print(loss_model_log)
                    log.write(loss_model_log+'\n')

                    dashed_line = '-'*80
                    head = f'{"Ground Truth":25s} | {"Prediction" :25s}| T/F'
                    predicted_result_log = f'{dashed_line}\n{head}\n{dashed_line}\n'

                    random_idx  = np.random.choice(range(len(label_top)), size= 5, replace=False)
                    label_concat = np.concatenate([np.asarray(label_top).reshape(1,-1), np.asarray(label_mid).reshape(1,-1), np.asarray(label_bot).reshape(1,-1)], axis=0).reshape(3,-1)
                    pred_concat = np.concatenate([np.asarray(pred_top_str).reshape(1,-1), np.asarray(pred_mid_str).reshape(1,-1), np.asarray(pred_bot_str).reshape(1,-1)], axis=0).reshape(3,-1)
                    
                    for i in random_idx:
                        label_sample = label_concat[:, i]
                        pred_sample = pred_concat[:, i]

                        gt_str = utils.str_combine(label_sample[0], label_sample[1], label_sample[2])
                        pred_str = utils.str_combine(pred_sample[0], pred_sample[1], pred_sample[2])
                        predicted_result_log += f'{gt_str:25s} | {pred_str:25s} | \t{str(pred_str == gt_str)}\n'
                    predicted_result_log += f'{dashed_line}'
                    print(predicted_result_log)
                    log.write(predicted_result_log+'\n')
                    
                # Stochastic weight averaging
                optimizer.update_swa()
                swa_count+=1
                if swa_count % 5 ==0:
                    optimizer.swap_swa_sgd()
                    torch.save(model.module.state_dict(), f'./models/{opt.experiment_name}/swa_{swa_count}.pth')

        if (n_epoch) % 5 ==0:
            torch.save(model.module.state_dict(), f'./models/{opt.experiment_name}/{n_epoch}.pth')


# ## main

# In[ ]:


os.makedirs(f'./models/{opt.experiment_name}', exist_ok=True)

# set seed
random.seed(opt.manualSeed)
np.random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
torch.cuda.manual_seed(opt.manualSeed)

# set GPU
cudnn.benchmark = True
cudnn.deterministic = True
opt.num_gpu = torch.cuda.device_count()

# if opt.num_gpu > 1:
#     print('-------- Use multi GPU setting --------')
#     opt.workers = opt.workers * opt.num_gpu
#     opt.batch_size = opt.batch_size * opt.num_gpu

train(opt)


# ----------------

# ## validation by visualization

# In[ ]:


import matplotlib.font_manager as fm
fm.get_fontconfig_fonts()
font_location = '/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf'
fontprop = fm.FontProperties(fname=font_location)
device = torch.device('cpu') 

class Valid_visualizer():
    def __init__(self, opt, model_path, val_path, visual_samples, device):
        self.opt = opt
        self.model_path = model_path
        self.val_path = val_path
        self.dataset = self._get_dataset()
        self.visual_samples = visual_samples
        self.device = device
        
    def _load_model(self):
        model = www_model.STR(self.opt, self.device)
        model.load_state_dict(torch.load(self.model_path))
        model.to(self.device)
        model.eval()
        return model
    
    def _get_dataset(self):
        val_list = os.listdir(self.val_path)
        val_dataset = []
        label = 'ㄱ'
        for val in val_list:
#             img = Image.open(f'./val/{val}').convert('RGB')
            val_dataset.append([os.path.join(self.val_path, val), label])
        return val_dataset
    
    def _get_valid_loader(self):

        test_streamer = utils.Dataset_streamer(self.dataset, resize_shape = (opt.imgH, opt.imgW), transformer=ToTensor())
#         _AlignCollate = utils.AlignCollate(imgH=self.opt.imgH, imgW=self.opt.imgW, keep_ratio_with_pad=True)
#         test_loader = DataLoader(test_streamer, batch_size = len(self.dataset), num_workers =0, collate_fn = _AlignCollate,)
        test_loader = DataLoader(test_streamer, batch_size = self.visual_samples, num_workers =0)
        return iter(test_loader)
    
    
    def valid_visualize(self):
        random.shuffle(self.dataset)
        test_loader_iter = self._get_valid_loader()
        image_tensor, label = next(test_loader_iter)
        model = self._load_model()
        output = model(input = image_tensor, text= ' ', is_train=False)
        pred_index = output.max(2)[1]
        pred_length = torch.IntTensor([opt.batch_max_length] * self.visual_samples).to(device)
        pred_decode = converter.decode(pred_index, pred_length)
        preds = []
        
        for pred in pred_decode:
            pred_temp = pred[ : pred.find('[s]')]
        #             pred_temp = join_jamos(pred_temp)
            preds.append(pred_temp)
        
        n_cols = 5
        n_rows = int(np.ceil(self.visual_samples/n_cols))
        last = self.visual_samples % n_cols
        fig, axes = plt.subplots(n_rows, n_cols)
        fig.set_size_inches((30, 30))
        i=0      
        for row in range(n_rows):
            for col in range(n_cols):
                axes[row][col].imshow(Image.open(self.dataset[i][0]))
                axes[row][col].set_xlabel(f'Prediction : {preds[i]}', fontproperties=fontprop, fontsize=30)
                i+=1
                if (row==n_rows-1) & (col==last-1):
                    break


# In[ ]:


vv = Valid_visualizer(opt, model_path = './models/www_0708/2/best_accuracy_91.602.pth', val_path = './val', visual_samples = 8, device= device)


# In[ ]:


vv.valid_visualize()


# In[ ]:




