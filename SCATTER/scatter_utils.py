import json
import pandas as pd
import os
import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.nn.init as init
# from torch.utils.data import *
import numpy as np
import time
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import easydict
import sys
import re
import six
import math
import torchvision.transforms as transforms
from jamo import h2j, j2hcj, j2h
import torch.nn.functional as F
from nltk.metrics.distance import edit_distance

sys.path.append('./Whatiswrong')
sys.path.append('./Scatter')
import utils

device = torch.device('cuda')


def validation(model, criterion, evaluation_loader, converter, opt):
    """ validation or evaluation """
    n_correct = 0
    norm_ED = 0
    length_of_data = 0
    infer_time = 0
    valid_loss_avg = Averager()
    
    ctc_criterion = criterion[0]
    scr_criterion = criterion[1]
    
    
    for i, (image_tensors, labels) in enumerate(evaluation_loader):
        batch_size = image_tensors.size(0)
        length_of_data = length_of_data + batch_size
        image = image_tensors.to(device)
        # For max length prediction
        length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device) # batch_max_length 숫자로 이루어진 bs 만큼의 벡터[27,27,27..]
        text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device) # bs * batch_max_length+1 사이즈의 0으로 채워진 텐서

        text_for_loss, length_for_loss = converter.encode(labels, batch_max_length=opt.batch_max_length)

        start_time = time.time()
        
#         if 'CTC' in opt.Prediction:
#             preds = model(image, text_for_pred)
#             forward_time = time.time() - start_time

#             # Calculate evaluation loss for CTC deocder.
#             preds_size = torch.IntTensor([preds.size(1)] * batch_size)
#             # permute 'preds' to use CTCloss format
#             cost = criterion(preds.log_softmax(2).permute(1, 0, 2), text_for_loss, preds_size, length_for_loss)

#             # Select max probabilty (greedy decoding) then decode index to character
#             _, preds_index = preds.max(2)
#             preds_index = preds_index.view(-1)
#             preds_str = converter.decode(preds_index.data, preds_size.data)

        scr_probs, ctc_prob = model(image, text_for_pred, is_train=False)
 
        target = text_for_loss[:, 1:]
    
        # ignore ctc loss 
#         input_lengths = torch.full(size = (ctc_prob.size(0),), fill_value= ctc_prob.size(1), dtype=torch.long)
#         output_lengths = torch.randint(low = 1, high = ctc_prob.size(1), size = (ctc_prob.size(0), ), dtype = torch.long)    
#         ctc_cost = ctc_criterion(ctc_prob.transpose(0,1), target, input_lengths, output_lengths)
#         valid_loss_avg.add(ctc_cost)
        
        # ignore other selective contexual decoder probs except for last one
        pred = scr_probs[1][:, :text_for_loss.shape[1] - 1, :] # pred.size = (batch_size, batch_max_length+1, n_classes)
          # without [GO] Symbol
        scr_cost = scr_criterion(pred.contiguous().view(-1, pred.shape[-1]), target.contiguous().view(-1))
        valid_loss_avg.add(scr_cost)
            
        # select max probabilty (greedy decoding) then decode index to character
        _, preds_index = pred.max(2)
        preds_str = converter.decode(preds_index, length_for_pred)
        labels = converter.decode(text_for_loss[:, 1:], length_for_loss)
        
        forward_time = time.time() - start_time
        infer_time += forward_time

        # calculate accuracy & confidence score
        preds_prob = F.softmax(pred, dim=2)
        preds_max_prob, _ = preds_prob.max(dim=2)
        confidence_score_list = []
        for gt, pred, pred_max_prob in zip(labels, preds_str, preds_max_prob):
#             if 'f' in opt.Prediction:
            gt = gt[:gt.find('[s]')]
            pred_EOS = pred.find('[s]')
            pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
            pred_max_prob = pred_max_prob[:pred_EOS]

            # To evaluate 'case sensitive model' with alphanumeric and case insensitve setting.
#             if opt.sensitive and opt.data_filtering_off:
#                 pred = pred.lower()
#                 gt = gt.lower()
#                 alphanumeric_case_insensitve = '0123456789abcdefghijklmnopqrstuvwxyz'
#                 out_of_alphanumeric_case_insensitve = f'[^{alphanumeric_case_insensitve}]'
#                 pred = re.sub(out_of_alphanumeric_case_insensitve, '', pred)
#                 gt = re.sub(out_of_alphanumeric_case_insensitve, '', gt)

            if pred == gt:
                n_correct += 1

            '''
            (old version) ICDAR2017 DOST Normalized Edit Distance https://rrc.cvc.uab.es/?ch=7&com=tasks
            "For each word we calculate the normalized edit distance to the length of the ground truth transcription."
            if len(gt) == 0:
                norm_ED += 1
            else:
                norm_ED += edit_distance(pred, gt) / len(gt)
            '''

            # ICDAR2019 Normalized Edit Distance
            if len(gt) == 0 or len(pred) == 0:
                norm_ED += 0
            elif len(gt) > len(pred):
                norm_ED += 1 - edit_distance(pred, gt) / len(gt)
            else:
                norm_ED += 1 - edit_distance(pred, gt) / len(pred)

            # calculate confidence score (= multiply of pred_max_prob)
            try:
                confidence_score = pred_max_prob.cumprod(dim=0)[-1]
            except:
                confidence_score = 0  # for empty pred case, when prune after "end of sentence" token ([s])
            confidence_score_list.append(confidence_score)
            # print(pred, gt, pred==gt, confidence_score)

    accuracy = n_correct / float(length_of_data) * 100
    norm_ED = norm_ED / float(length_of_data)  # ICDAR2019 Normalized Edit Distance

    return valid_loss_avg.val(), accuracy, norm_ED, preds_str, confidence_score_list, labels, infer_time, length_of_data

class Averager(object):
    
    def __init__(self):
        self.reset()
    
    def add(self, v):
        count = v.data.numel() #number of elements
        v = v.data.sum()
        self.n_count += count
        self.sum += v
        
    def reset(self):
        self.n_count = 0
        self.sum = 0
    
    def val(self):
        res = 0
        if self.n_count !=0:
            res = self.sum / float(self.n_count)
        return res