import os
import random
import torch
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.optim as optim
import torch.utils.data
import numpy as np
import six
import math
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.data import *
import torch.nn.functional as F

# class mySequential(nn.Sequential):
#     def forward(self, *inputs):
#         for module in self._modules.values():
#             if type(inputs) == tuple:
#                 inputs = module(*inputs)
# #             else:
# #                 inputs = module(inputs)
# #         print(inputs)
#         return inputs
    
    
# class SCR_Blocks(torch.nn.Module):
#     def __init__(self, input_size, hidden_size, output_size, num_classes, device, batch_max_length, n_blocks):
#         super(SCR_Blocks, self).__init__()
#         self.SCR_blocks = self._make_blocks(input_size, hidden_size, output_size, num_classes, device, batch_max_length, n_blocks)
        
#     def forward(self, input_dic):
#         V,  Probs_list, label, is_train = input_dic
#         Prob_list = self.SCR_blocks(V, Probs_list, label, is_train)
#         return Prob_list
        
        
#     def _make_blocks(self,input_size, hidden_size, output_size, num_classes, device, batch_max_length, n_blocks):
#         layers = []
#         for i in range(n_blocks-1):
#             layers.append(Selective_Contextual_refinement_block(input_size, hidden_size, output_size, num_classes, 
#                                                                 device, batch_max_length, decoder_fix = False ))
            
#         layers.append(Selective_Contextual_refinement_block(input_size, hidden_size, output_size, num_classes, 
#                                                             device, batch_max_length, decoder_fix = True))
        
#         return mySequential(*layers)
        
            
class Selective_Contextual_refinement_block(torch.nn.Module):
    
    def __init__(self, input_size, hidden_size, output_size, vertical_num_classes, device, batch_max_length, decoder_fix):
        super(Selective_Contextual_refinement_block, self).__init__()
        self.BiLSTM = torch.nn.LSTM(input_size, hidden_size, num_layers=2, bidirectional = True, batch_first = True)
        self.linear = torch.nn.Linear(hidden_size *2, output_size) # V 와 채널수 맞추기위함
        self.selective_decoder = Selective_decoder(input_size*2, vertical_num_classes, device, batch_max_length )
        self.decoder_fix = decoder_fix

    
    def forward(self, H, V, Probs_list, label_list, is_train):
        self.BiLSTM.flatten_parameters()
        H, _ = self.BiLSTM(H)
        H = self.linear(H)
        D = torch.cat([H, V], dim= -1)
        
        if (is_train==True) & (self.decoder_fix == False) :
            Probs = self.selective_decoder(D, label_list, is_train)
#             return Probs, H

            Probs_list.append(Probs)
            return H, Probs_list
        
        elif self.decoder_fix ==True:
            Probs = self.selective_decoder(D, label_list, is_train)
#             return Probs, _

            Probs_list.append(Probs)
            return Probs_list
        
        elif (is_train==False) & (self.decoder_fix ==False):
#             return _, H
            return H, Probs_list
        
#         else: # is_train ==False & decoder_fix ==True :
#             Probs = self.selective_decoder(D, label, is_train)
#             return Probs, _
       

    
class Selective_decoder(torch.nn.Module):
    def __init__(self, input_size, vertical_num_classes, device, batch_max_length):
        super(Selective_decoder, self).__init__()
        self.attention_1d = torch.nn.Linear(in_features = input_size, out_features = input_size)
        self.device = device
#         self.lstm = torch.nn.LSTM(input_size, 1, batch_first= True)
        self.attention_2d = Attention(input_size, vertical_num_classes, device=self.device, batch_max_length = batch_max_length )
#         self.num_classes = num_classes
        
        
    def forward(self, D, label_list , is_train):
#         D = self.attention_1d(D)
        attention_map = self.attention_1d(D)
#         attention_map = torch.sigmoid(D)
        D_prime = attention_map * D
        probs = self.attention_2d(D_prime, label_list, is_train)
        return probs
        
        
class Attention(torch.nn.Module):
    def __init__(self, input_size, vertical_num_classes, device, batch_max_length): #vertical_num_classes order : bot, mid, high
        super(Attention, self).__init__()
        self.device = device
        self.vertical_num_classes = vertical_num_classes
        self.attention_cell = AttentionCell(input_size, input_size, vertical_num_classes, device)
        self.input_size = input_size
        self.generator = torch.nn.Linear(input_size, vertical_num_classes[-1])
        self.batch_max_length  = batch_max_length
           
    def _char_to_onehot(self, input_char, onehot_dim):
        input_char = input_char.unsqueeze(1)
        batch_size = input_char.size(0)
        onehot = torch.FloatTensor(batch_size, onehot_dim).zero_().to(self.device)
        onehot = onehot.scatter_(dim=1, index = input_char, value=1)
        return onehot
    
    def forward(self, D_prime, label_list, is_train):
        batch_size = D_prime.size(0)
        num_steps = self.batch_max_length +1 # for additional [EOS]

        output_hiddens = torch.FloatTensor(batch_size, num_steps, self.input_size).fill_(0).to(self.device)
        
        hidden = (torch.FloatTensor(batch_size, self.input_size).fill_(0).to(self.device),
                  torch.FloatTensor(batch_size, self.input_size).fill_(0).to(self.device))
        
        # For bottom 
        if len(label_list)==1:
            bot_text = label_list[0]

            if is_train:
                for i in range(num_steps):
                    bot_text_onehots = self._char_to_onehot(bot_text[:,i], onehot_dim = self.vertical_num_classes[0])
                    hidden, alpha = self.attention_cell(hidden, D_prime, [bot_text_onehots])
                    output_hiddens[:, i, :] = hidden[0]
                probs = self.generator(output_hiddens)

            else:
                targets = torch.LongTensor(batch_size).fill_(0).to(self.device)
                probs = torch.FloatTensor(batch_size, num_steps, self.vertical_num_classes[0]).fill_(0).to(self.device)

                for i in range(num_steps):
                    bot_text_onehots = self._char_to_onehot(targets, onehot_dim = self.vertical_num_classes[0])
                    hidden, alpha = self.attention_cell(hidden, D_prime, [bot_text_onehots])
                    probs_step = self.generator(hidden[0])
                    probs[:, i, :] = probs_step
                    _, next_input = probs_step.max(1)
                    targets = next_input

            return probs
        
        # For middle
        elif len(label_list)==2:
            bot_text, mid_text = label_list
            
            if is_train:
                for i in range(num_steps):
                    bot_text_onehots = self._char_to_onehot(bot_text[:,i], onehot_dim = self.vertical_num_classes[0])
                    mid_text_onehots = self._char_to_onehot(mid_text[:,i], onehot_dim = self.vertical_num_classes[1])
                    hidden, alpha = self.attention_cell(hidden, D_prime, [bot_text_onehots, mid_text_onehots])
                    output_hiddens[:, i, :] = hidden[0]
                probs = self.generator(output_hiddens)

            else:
                targets = torch.LongTensor(batch_size).fill_(0).to(self.device)
                probs = torch.FloatTensor(batch_size, num_steps, self.vertical_num_classes[1]).fill_(0).to(self.device)

                for i in range(num_steps):
                    bot_text_onehots = self._char_to_onehot(bot_text[:i], onehot_dim = self.vertical_num_classes[0])
                    mid_text_onehots = self._char_to_onehot(targets, onehot_dim = self.vertical_num_classes[1])
                    hidden, alpha = self.attention_cell(hidden, D_prime, [bot_text_onehots, mid_text_onehots])
                    probs_step = self.generator(hidden[0])
                    probs[:, i, :] = probs_step
                    _, next_input = probs_step.max(1)
                    targets = next_input

            return probs
        
        # For top
        elif len(label_list)==3:
            bot_text, mid_text, top_text = label_list
            
            if is_train:
                for i in range(num_steps):
                    bot_text_onehots = self._char_to_onehot(bot_text[:,i], onehot_dim = self.vertical_num_classes[0])
                    mid_text_onehots = self._char_to_onehot(mid_text[:,i], onehot_dim = self.vertical_num_classes[1])
                    top_text_onehots = self._char_to_onehot(top_text[:,i], onehot_dim = self.vertical_num_classes[2])
                    hidden, alpha = self.attention_cell(hidden, D_prime, [bot_text_onehots, mid_text_onehots, top_text_onehots])
                    output_hiddens[:, i, :] = hidden[0]
                probs = self.generator(output_hiddens)

            else:
                targets = torch.LongTensor(batch_size).fill_(0).to(self.device)
                probs = torch.FloatTensor(batch_size, num_steps, self.vertical_num_classes[2]).fill_(0).to(self.device)

                for i in range(num_steps):
                    bot_text_onehots = self._char_to_onehot(bot_text[:,i], onehot_dim = self.vertical_num_classes[0])
                    mid_text_onehots = self._char_to_onehot(mid_text[:,i], onehot_dim = self.vertical_num_classes[1])
                    top_text_onehots = self._char_to_onehot(targets, onehot_dim = self.vertical_num_classes[2])
                    hidden, alpha = self.attention_cell(hidden, D_prime, [bot_text_onehots, mid_text_onehots])
                    probs_step = self.generator(hidden[0])
                    probs[:, i, :] = probs_step
                    _, next_input = probs_step.max(1)
                    targets = next_input

            return probs
        
        
        
            

class AttentionCell(torch.nn.Module):
    def __init__(self, input_size, output_size, vertical_num_classes, device ):
        super(AttentionCell, self).__init__()
        self.vertical_num_classes = vertical_num_classes
        self.i2h = torch.nn.Linear(input_size, output_size, bias= True)
        self.h2h = torch.nn.Linear(input_size, output_size, bias=False)
        self.score = torch.nn.Linear(output_size, 1, bias=False)
        ######
        sum_classes = input_size
        for vertical_class in vertical_num_classes:
            sum_classes += vertical_class
        ######
        self.lstm = torch.nn.LSTMCell(sum_classes, output_size)
        
    def forward(self, hidden, D_prime, vertical_text_list):
        
        i2h_output = self.i2h(D_prime)
        h2h_output = self.h2h(hidden[0])
        e_t = self.score(torch.tanh(i2h_output + (h2h_output.unsqueeze(1))))
        a_t = F.softmax(e_t, dim=1)
        g_t = torch.bmm(a_t.permute(0, 2, 1), D_prime).squeeze(1) 
        ######
        concat = [g_t]
#         print(f'g_t shape : {g_t.shape}')
        for vertical_text in vertical_text_list:
#             print(f'vertical_text shape : {vertical_text.shape}')
            concat.append(vertical_text)
        s_t = self.lstm(torch.cat(concat, dim=-1), hidden)
        ######
        return s_t, a_t
    
    