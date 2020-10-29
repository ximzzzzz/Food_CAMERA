import logging
from ts.torch_handler.base_handler import BaseHandler
from albumentations.pytorch import ToTensor
import json
import pandas as pd
import os
import random
import torch
import torchvision.transforms as transforms
import math
import numpy as np
from PIL import Image
import io

class RecognitionHandler(BaseHandler):
    
    def __init__(self):
        super().__init__()
        self.resize_H = 64
        self.resize_W = 256
        self.normalize_pad = NormalizePAD((3, self.resize_H, self.resize_W))        
        self.transformer = ToTensor()
        self.top_char = ' !"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZㄱㄴㄷㄹㅁㅂㅅㅇㅈㅊㅋㅌㅍㅎㄲㄸㅃㅆㅉ',
        self.mid_char = ' ㅏㅑㅓㅕㅗㅛㅜㅠㅡㅣㅐㅒㅔㅖㅘㅙㅚㅝㅞㅟㅢ',
        self.bot_char = ' ㄱㄴㄷㄹㅁㅂㅅㅇㅈㅊㅋㅌㅍㅎㄲㄸㅃㅆㅉㄳㄵㄶㄺㄻㄼㄽㄾㄿㅀㅄ',
        self.device = torch.device('cuda')
        
    def preprocess_one_image(self, req):
        
        image = req.get('data')
        if image is None:
            image = req.get('body')
        image = Image.open(io.BytesIO(image)).convert('RGB')
        
        orig_H = np.array(image).shape[0]
        orig_W = np.array(image).shape[1]
        ratio = orig_W / float(orig_H)
        
        if math.ceil(self.resize_H * ratio) > self.resize_W:
            resize_W = self.resize_W
        else :
            resize_W = math.ceil(self.resize_H * ratio)
            
        image = np.array(image.resize((resize_W, self.resize_H), Image.BICUBIC))
        image = self.normalize_pad(image)
        image = tensor2im(image)

        return self.transformer(**{'image' : image})['image']
        
            
    def preprocess(self, requests):
        images = [self.preprocess_one_image(req) for req in requests]
        images = torch.cat(images).unsqueeze(0).to(self.device)
        return images
        
    def inference(self, images):
        batch_size = images.shape[0]
        dummy_text = torch.IntTensor(batch_size, 26).fill_(0)
        pred_top, pred_mid, pred_bot = self.model(images, dummy_text, dummy_text, dummy_text)

        return (pred_top, pred_mid, pred_bot) 
    
    def postprocess(self, preds):
        _, pred_top_index = preds[0].max(2)
        _, pred_mid_index = preds[1].max(2)
        _, pred_bot_index = preds[2].max(2)
        
        attn_converter_top = AttnLabelConverter(self.top_char, self.device)
        attn_converter_mid = AttnLabelConverter(self.mid_char, self.device)
        attn_converter_bot = AttnLabelConverter(self.bot_char, self.device)
        
        decode_top = attn_converter_top.decode(pred_top_index, torch.FloatTensor(batch_size, 26, attn_converter_top.n_cls+2))
        decode_mid = attn_converter_mid.decode(pred_mid_index, torch.FloatTensor(batch_size, 26, attn_converter_mid.n_cls+2))
        decode_bot = attn_converter_bot.decode(pred_bot_index, torch.FloatTensor(batch_size, 26, attn_converter_bot.n_cls+2))
        
        batch_size = len(decode_top)
        recognition_list = []
        for i in range(batch_size):
            str_combined = str_combine(decode_top[i], decode_mid[i], decode_bot[i])
            if not str_combined=='':
                recognition_list.append(utils.str_combine(decode_top[i], decode_mid[i], decode_bot[i]))
                
        return recognition_list
    
class NormalizePAD(object):
    
    def __init__(self, max_size, PAD_type='right'):
        self.toTensor = transforms.ToTensor()
        self.max_size = max_size
        self.max_width_half = math.floor(max_size[2]/2)
        self.PAD_type = PAD_type
        
    def __call__(self, img):
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5) #빼고 나누는 연산을 inplace
        c, h, w = img.size()
        Pad_img = torch.FloatTensor(*self.max_size).fill_(0)
        Pad_img[:,:, :w] = img # right pad
        if self.max_size[2] != w:
            Pad_img[:, :, w:] = img[:, :, w - 1].unsqueeze(2).expand(c, h, self.max_size[2] - w)
            
#         return np.asarray(Pad_img)
        return Pad_img


class AttnLabelConverter(object):
    
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
        # additional +1 for [GO] at first step. batch_text is padded with [GO]  token after [s] token.
        batch_text = torch.LongTensor(len(text), batch_max_length + 1).fill_(0)
        for i, t in enumerate(text):
            text = list(t)
            text.append('[s]')
            text = [self.dict[char] for char in text]
            batch_text[i][1 : 1+len(text)] = torch.LongTensor(text)
        return (batch_text.to(self.device ), torch.IntTensor(length).to(self.device))
#         return (batch_text, torch.IntTensor(length))
    
    
    def decode(self, text_index, length):
        texts = []
        for index, i in enumerate(length):
            text = ''.join([self.character[i] for i in text_index[index, :]])
            texts.append(text)
            
        return texts

    
def str_combine(decode_top, decode_mid, decode_bot):
        
    decode_top = decode_top[:decode_top.find('[s]')]
    decode_mid = decode_mid[:decode_mid.find('[s]')]
    decode_bot = decode_bot[:decode_bot.find('[s]')]
    
    decode_top_ = list(decode_top)
    total_length = len(decode_top_)
    
    decode_mid_ = [' ']*total_length
    decode_mid_[:len(decode_mid[:total_length])] = list(decode_mid)[:total_length]
    
    decode_bot_ = [' ']*total_length
    decode_bot_[:len(decode_bot[:total_length])] = list(decode_bot)[:total_length]
    
    combine_arr = np.array([decode_top_, decode_mid_, decode_bot_]).reshape(3,-1)
#     print(combine_arr)
    combine_res = ''
    for i in range(combine_arr.shape[1]):
        char = combine_arr[:,i]
        if ((char[0] in 'ㄱㄴㄷㄹㅁㅂㅅㅇㅈㅊㅋㅌㅍㅎㄲㄸㅃㅆㅉ') & (char[1]==' ')) :
            one_char = ''
            
        elif ((char[0] in ' !"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ') & ((char[1]!=' ') | (char[2]!=' '))):
            one_char = char[0]
            
        else :
            one_char = join_jamos(char).strip()
        combine_res = combine_res+one_char
    return combine_res


def tensor2im(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor.float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)

def join_jamos(s, ignore_err=True):

    last_t = 0
    queue = []
    new_string = ""

    def flush(n=0):
        new_queue = []
        while len(queue) > n:
            new_queue.append(queue.pop())
        if len(new_queue) == 1:
            if not ignore_err:
                raise ValueError(f"invalid jamo character: {new_queue[0]}")
            result = new_queue[0]
        elif len(new_queue) >= 2:
            try:
                result = join_jamos_char(*new_queue)
            except (ValueError, KeyError):
                # Invalid jamo combination
                if not ignore_err:
                    raise ValueError(f"invalid jamo characters: {new_queue}")
                result = "".join(new_queue)
        else:
            result = None
        return result

    for c in s:
        if c not in CHARSET:
            if queue:
                new_c = flush() + c
            else:
                new_c = c
            last_t = 0
        else:
            t = get_jamo_type(c)
            new_c = None
            if t & FINAL == FINAL:
                if not (last_t == MEDIAL):
                    new_c = flush()
            elif t == INITIAL:
                new_c = flush()
            elif t == MEDIAL:
                if last_t & INITIAL == INITIAL:
                    new_c = flush(1)
                else:
                    new_c = flush()
            last_t = t
            queue.insert(0, c)
        if new_c:
            new_string += new_c
    if queue:
        new_string += flush()
    return new_string