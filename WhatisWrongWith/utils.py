import os
import sys
import re
import six
import math
import lmdb
import torch
import time
sys.path.append('./Scatter')
sys.path.append('../Scatter')
import augs
from natsort import natsorted
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
from torch.utils.data import Dataset, ConcatDataset, Subset
from torch._utils import _accumulate
import torchvision.transforms as transforms
from itertools import islice, cycle
import cv2
import torch.nn.functional as F
from nltk.metrics.distance import edit_distance


device = torch.device('cuda')
# device = torch.device('cpu')

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
        one_char = join_jamos(char).strip()
        combine_res = combine_res+one_char
    return combine_res

def SaveDir_maker(base_model, base_model_dir = './models', ):
    trial_count = 0
    mon = str(time.localtime().tm_mon) if len(str(time.localtime().tm_mon)) ==2 else '0'+str(time.localtime().tm_mon)
    day = str(time.localtime().tm_mday) if len(str(time.localtime().tm_mday)) ==2 else '0'+str(time.localtime().tm_mday)
    directory = f'{base_model}_{mon}{day}/{trial_count}'
    empty_flag = True
    while empty_flag :
        if (os.path.exists(os.path.join(base_model_dir, directory))) and (os.path.isfile(os.path.join(base_model_dir, directory, 'best_accuracy.pth'))):
            trial_count+=1
            directory = f'{base_model}_{mon}{day}/{trial_count}'
        else: 
            empty_flag=False
    return directory


def get_transform():
    transform = transforms.Compose([
                                transforms.ColorJitter(brightness=(0.0, 0.3), contrast=(0.0,0.3), saturation = (0.0,0.2), hue= (0.0,0.2)),
                                transforms.RandomAffine(degrees = 0, translate = (0.3, 0.3)),
                                transforms.RandomPerspective(distortion_scale=0.3, p=0.5, interpolation=3, fill=0)
                                ])
    return transform


def tensor2im(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor.float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)


class Dataset_streamer(Dataset):
    
    def __init__(self, dataset, resize_shape = (32, 200), input_channel = 3, transformer=None):
        self.dataset = dataset
        self.transformer = transformer
        self.resize_H = resize_shape[0]
        self.resize_W = resize_shape[1]
        self.toTensor = transforms.ToTensor()
        self.normalize_pad = NormalizePAD((input_channel, self.resize_H, self.resize_W))

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_path, label = self.dataset[idx]
        image = Image.open(img_path).convert('RGB')
        
        #세로형 이미지 가로로 잘라옮기기
        img_arr = np.asarray(image)
        h, w, c = img_arr.shape
        if h > w:
            char_length = len(label)
            try :
                each_char_height = int(np.ceil(h /char_length))
                new_shape = np.zeros((each_char_height ,w * char_length, 3))
                for i in range(char_length):
                    cropped = img_arr[i*each_char_height : (i+1) * each_char_height, :, : ]
                    height = cropped.shape[0]
                    new_shape[ : height, i*w : (i+1)*w, : ] = cropped

                image = Image.fromarray((new_shape*255).astype(np.uint8))
                
            except :
                img_path, label = self.dataset[idx+1]
                image = Image.open(img_path).convert('RGB')
        
        # normalize with padding
#         img_tensor = self.toTensor(image)
#         orig_H = img_tensor.size(1)
#         orig_W = img_tensor.size(2)
        orig_H = img_arr.shape[0]
        orig_W = img_arr.shape[1]
        ratio = orig_W / float(orig_H)
        
        if math.ceil(self.resize_H * ratio) > self.resize_W:
            resize_W = self.resize_W
        else :
            resize_W = math.ceil(self.resize_H * ratio)
        
        image = np.array(image.resize((resize_W, self.resize_H), Image.BICUBIC))
        image = self.normalize_pad(image)
        image = tensor2im(image)


        if self.transformer:
            return (self.transformer(**{'image' : image, 'label' : label })['image'], label)
        
        else:
            return (image, label)


def make_str(label):
    string = ''
    for lab in label:
        string+=lab
    return string

class CustomDataset_jamo(Dataset):
    
    def __init__(self, dataset, resize_shape = (32, 200), input_channel = 3, transformer=None):
        self.dataset = dataset
        self.transformer = transformer
        self.resize_H = resize_shape[0]
        self.resize_W = resize_shape[1]
        self.toTensor = transforms.ToTensor()
        self.normalize_pad = NormalizePAD((input_channel, self.resize_H, self.resize_W))

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_path, label_top, label_mid, label_bot = self.dataset[idx]
        image = Image.open(img_path).convert('RGB')

        img_arr = np.asarray(image)
        h, w, c = img_arr.shape
        if h > w:
            img_path, label_top, label_mid, label_bot = self.dataset[idx-1]
            image = Image.open(img_path).convert('RGB')
        
        orig_H = img_arr.shape[0]
        orig_W = img_arr.shape[1]
        ratio = orig_W / float(orig_H)
        
        if math.ceil(self.resize_H * ratio) > self.resize_W:
            resize_W = self.resize_W
        else :
            resize_W = math.ceil(self.resize_H * ratio)
        
        image = np.array(image.resize((resize_W, self.resize_H), Image.BICUBIC))
        image = self.normalize_pad(image)
        image = tensor2im(image)
        
        label_top = make_str(label_top)
        label_mid = make_str(label_mid)
        label_bot = make_str(label_bot)

        if self.transformer:
            return (self.transformer(**{'image' : image })['image'], label_top, label_mid, label_bot)
        
        else:
            return (image, label_top, label_mid, label_bot)
    
    
class AttnLabelConverter(object):
    
    def __init__(self, character):
        list_token = ['[GO]', '[s]']
        list_character = list(character)
        self.character = list_token + list_character
        
        self.dict = {}
        for i, char in enumerate(self.character):
            self.dict[char] = i
    
    def encode(self, text, batch_max_length = 25):
        length = [len(s) + 1 for s in text] # +1 for [s]
        batch_max_length +=1
        # additional +1 for [GO] at first step. batch_text is padded with [GO]  token after [s] token.
        batch_text = torch.LongTensor(len(text), batch_max_length + 1).fill_(0)
        for i, t in enumerate(text):
#             print('t', t)
            text = list(t)
#             print('text', text)
            text.append('[s]')
#             print('before encoded', text) 
            text = [self.dict[char] for char in text]
#             print('encoded' ,text)
            batch_text[i][1 : 1+len(text)] = torch.LongTensor(text)
        return (batch_text.to(device), torch.IntTensor(length).to(device))
#         return (batch_text, torch.IntTensor(length))
    
    
    def decode(self, text_index, length):
        texts = []
        for index, i in enumerate(length):
            text = ''.join([self.character[i] for i in text_index[index, :]])
            texts.append(text)
            
        return texts
    
    
    
class CTCLabelConverter(object):
    """ Convert between text-label and text-index """

    def __init__(self, character):
        # character (str): set of the possible characters.
        dict_character = list(character)

        self.dict = {}
        for i, char in enumerate(dict_character):
            # NOTE: 0 is reserved for 'CTCblank' token required by CTCLoss
            self.dict[char] = i + 1

        self.character = ['[CTCblank]'] + dict_character  # dummy '[CTCblank]' token for CTCLoss (index 0)

    def encode(self, text, batch_max_length=25):
        """convert text-label into text-index.
        input:
            text: text labels of each image. [batch_size]
            batch_max_length: max length of text label in the batch. 25 by default
        output:
            text: text index for CTCLoss. [batch_size, batch_max_length]
            length: length of each text. [batch_size]
        """
        length = [len(s) for s in text]

        # The index used for padding (=0) would not affect the CTC loss calculation.
        batch_text = torch.LongTensor(len(text), batch_max_length).fill_(0)
        for i, t in enumerate(text):
            text = list(t)
            text = [self.dict[char] for char in text]
            batch_text[i][:len(text)] = torch.LongTensor(text)
        return (batch_text.to(device), torch.IntTensor(length).to(device))

    def decode(self, text_index, length):
        """ convert text-index into text-label. """
        texts = []
        for index, l in enumerate(length):
            t = text_index[index, :]

            char_list = []
            for i in range(l):
                if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):  # removing repeated characters and blank.
                    char_list.append(self.character[t[i]])
            text = ''.join(char_list)

            texts.append(text)
        return texts
    

    
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
        

class ResizeNormalize(object):

    def __init__(self, size, interpolation=Image.BICUBIC):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img
        
class AlignCollate(object):
    
    def __init__(self,  imgH = 193, imgW = 370, keep_ratio_with_pad = True, ):
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio_with_pad = keep_ratio_with_pad

        
    def __call__(self, batch):
#         if self.is_train :         
        batch = filter(lambda x : x is not None, batch)
        images, labels = zip(*batch)
#         else:
#             images = next(iter(batch))
#         print(images)
        if self.keep_ratio_with_pad :
            resized_max_w = self.imgW
            input_channel = 3 
            transform = NormalizePAD((input_channel, self.imgH, resized_max_w))
            
            resized_images = []
            for image in images:
                w, h = image.size
                ratio = w / float(h)
                if math.ceil(self.imgH * ratio) > self.imgW:
                    resized_w = self.imgW
                else:
                    resized_w = math.ceil(self.imgH * ratio)
                
                resized_image = image.resize((resized_w, self.imgH), Image.BICUBIC)
                resized_images.append(transform(resized_image))
                
            image_tensors = torch.cat([t.unsqueeze(0) for t in resized_images], 0)
        
        else:
            transform = ResizeNormalize((self.imgW, self.imgH))
            image_tensors = [transform(image) for image in images]
            image_tensors = torch.cat([t.unsqueeze(0) for t in image_tensors], 0)
        
        return image_tensors, labels
    
    
def validation(model, criterion, evaluation_loader, converter, opt):
    """ validation or evaluation """
    n_correct = 0
    norm_ED = 0
    length_of_data = 0
    infer_time = 0
    valid_loss_avg = Averager()

    for i, (image_tensors, labels) in enumerate(evaluation_loader):
        batch_size = image_tensors.size(0)
        length_of_data = length_of_data + batch_size
        image = image_tensors.to(device)

        length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
#         text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)
        text_for_pred = torch.FloatTensor(batch_size, opt.batch_max_length +1, len(opt.character)+2)

        text_for_loss, length_for_loss = converter.encode(labels, batch_max_length=opt.batch_max_length)
        
        onehot = torch.FloatTensor(batch_size, opt.batch_max_length+2, len(opt.character)+2).zero_().to(device)
        text_for_loss = onehot.scatter(dim = 2, index = text_for_loss.unsqueeze(2).to(device), value = 1 ) #(bs, batch_max_length, num_characters)
        
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

        preds = model(image, text_for_pred, is_train=False)
        forward_time = time.time() - start_time

        preds = preds[:, :text_for_loss.shape[1] - 1, :]
        target = text_for_loss[:, 1:]  # without [GO] Symbol
#         cost = criterion(preds.contiguous().view(-1, preds.shape[-1]), target.contiguous().view(-1))
        cost = criterion(preds.contiguous().view(-1, preds.shape[-1]), target.contiguous().view(-1, target.shape[-1]))

        # select max probabilty (greedy decoding) then decode index to character
        _, preds_index = preds.max(2)
        preds_str = converter.decode(preds_index, length_for_pred)
#         labels = converter.decode(text_for_loss[:, 1:], length_for_loss)
        labels = converter.decode(text_for_loss[:, 1:].max(2)[1], length_for_loss)

        infer_time += forward_time
        valid_loss_avg.add(cost)

        # calculate accuracy & confidence score
        preds_prob = F.softmax(preds, dim=2)
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



def validation_jamo(model, criterion, evaluation_loader, top_converter, middle_converter, bottom_converter, opt):
    """ validation or evaluation """
    n_correct = 0
    norm_ED = 0
    length_of_data = 0
    infer_time = 0
    valid_loss_avg = Averager()

    for i, (image_tensors, top, mid, bot) in enumerate(evaluation_loader):
        batch_size = image_tensors.size(0)
        length_of_data = length_of_data + batch_size
        image = image_tensors.to(device)

        length_for_pred_top = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
        text_for_pred_top = torch.FloatTensor(batch_size, opt.batch_max_length +1, opt.top_n_cls+2)
        text_for_loss_top, length_for_loss_top = top_converter.encode(top, batch_max_length=opt.batch_max_length)
        
        length_for_pred_mid = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
        text_for_pred_mid = torch.FloatTensor(batch_size, opt.batch_max_length +1, opt.middle_n_cls+2)
        text_for_loss_mid, length_for_loss_mid = middle_converter.encode(mid, batch_max_length=opt.batch_max_length)
        
        length_for_pred_bot = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
        text_for_pred_bot = torch.FloatTensor(batch_size, opt.batch_max_length +1, opt.bottom_n_cls+2)
        text_for_loss_bot, length_for_loss_bot = bottom_converter.encode(bot, batch_max_length=opt.batch_max_length)
        
#         onehot = torch.FloatTensor(batch_size, opt.batch_max_length+2, len(opt.top_n_cls)+2).zero_().to(device)
#         text_for_loss = onehot.scatter(dim = 2, index = text_for_loss_top.unsqueeze(2).to(device), value=1) #(bs, batch_max_length, num_characters)
        
        start_time = time.time()

        pred_top, pred_mid, pred_bot = model(image, text_for_pred_top, text_for_pred_mid, text_for_pred_bot, is_train=False)
        forward_time = time.time() - start_time

        pred_top = pred_top[:, :text_for_loss_top.shape[1] -1, :]
        pred_mid = pred_mid[:, :text_for_loss_mid.shape[1] -1, :]
        pred_bot = pred_bot[:, :text_for_loss_bot.shape[1] -1, :]
        target_top = text_for_loss_top[:, 1:]  # without [GO] Symbol
        target_mid = text_for_loss_mid[:, 1:] 
        target_bot = text_for_loss_bot[:, 1:]
#         cost = criterion(preds.contiguous().view(-1, preds.shape[-1]), target.contiguous().view(-1))
        cost_top = criterion(pred_top.contiguous().view(-1, pred_top.shape[-1]), target_top.contiguous().view(-1))
        cost_mid = criterion(pred_mid.contiguous().view(-1, pred_mid.shape[-1]), target_mid.contiguous().view(-1))
        cost_bot = criterion(pred_bot.contiguous().view(-1, pred_bot.shape[-1]), target_bot.contiguous().view(-1))
        
        cost = cost_top + cost_mid + cost_bot

        # select max probabilty (greedy decoding) then decode index to character
        _, pred_top_index = pred_top.max(2)
        _, pred_mid_index = pred_mid.max(2)
        _, pred_bot_index = pred_bot.max(2)
        
        pred_top_str = top_converter.decode(pred_top_index, length_for_pred_top)
        pred_mid_str = middle_converter.decode(pred_mid_index, length_for_pred_mid)
        pred_bot_str = bottom_converter.decode(pred_bot_index, length_for_pred_bot)
        
        label_top = top_converter.decode(text_for_loss_top[:, 1:], length_for_loss_top)
        label_mid = middle_converter.decode(text_for_loss_mid[:, 1:], length_for_loss_mid)
        label_bot = bottom_converter.decode(text_for_loss_bot[:, 1:], length_for_loss_bot)
        
#         labels = converter.decode(text_for_loss[:, 1:].max(2)[1], length_for_loss)

        infer_time += forward_time
        valid_loss_avg.add(cost)

        # calculate accuracy & confidence score
#         preds_prob_top = F.softmax(pred_top, dim=2)
        
#         preds_max_prob, _ = preds_prob.max(dim=2)
#         confidence_score_list = []
        for gt_top, gt_mid, gt_bot, pred_top_, pred_mid_, pred_bot_ in zip(label_top, label_mid, label_bot, pred_top_str, pred_mid_str, pred_bot_str):

            gt_top = gt_top[:gt_top.find('[s]')]
            gt_mid = gt_mid[:gt_mid.find('[s]')]
            gt_bot = gt_bot[:gt_bot.find('[s]')]
            
            pred_top_ = pred_top_[:pred_top_.find('[s]')]  # prune after "end of sentence" token ([s])
            pred_mid_ = pred_mid_[:pred_mid_.find('[s]')]
            pred_bot_ = pred_bot_[:pred_bot_.find('[s]')]
#             pred_max_prob = pred_max_prob[:pred_EOS]

            # To evaluate 'case sensitive model' with alphanumeric and case insensitve setting.
#             if opt.sensitive and opt.data_filtering_off:
#                 pred = pred.lower()
#                 gt = gt.lower()
#                 alphanumeric_case_insensitve = '0123456789abcdefghijklmnopqrstuvwxyz'
#                 out_of_alphanumeric_case_insensitve = f'[^{alphanumeric_case_insensitve}]'
#                 pred = re.sub(out_of_alphanumeric_case_insensitve, '', pred)
#                 gt = re.sub(out_of_alphanumeric_case_insensitve, '', gt)
            
            
            '''
            (old version) ICDAR2017 DOST Normalized Edit Distance https://rrc.cvc.uab.es/?ch=7&com=tasks
            "For each word we calculate the normalized edit distance to the length of the ground truth transcription."
            if len(gt) == 0:
                norm_ED += 1
            else:
                norm_ED += edit_distance(pred, gt) / len(gt)
            '''
            
            for gt, pred in zip([gt_top, gt_mid, gt_bot], [pred_top_, pred_mid_, pred_bot_]):
                #str_maker로 합친 후  gt와 비교
#                 print(f' gt {gt}')
#                 print(f' pred {pred}')
                if gt==pred:
                    n_correct +=1
                
                # ICDAR2019 Normalized Edit Distance
                if len(gt) == 0 or len(pred) == 0:
                    norm_ED += 0
                elif len(gt) > len(pred):
                    norm_ED += 1 - edit_distance(pred, gt) / len(gt)
                else:
                    norm_ED += 1 - edit_distance(pred, gt) / len(pred)

#             # calculate confidence score (= multiply of pred_max_prob)
#             try:
#                 confidence_score = pred_max_prob.cumprod(dim=0)[-1]
#             except:
#                 confidence_score = 0  # for empty pred case, when prune after "end of sentence" token ([s])
#             confidence_score_list.append(confidence_score)
            # print(pred, gt, pred==gt, confidence_score)

    accuracy = (n_correct/3) / float(length_of_data) * 100
    norm_ED = norm_ED / float(length_of_data)  # ICDAR2019 Normalized Edit Distance

    return valid_loss_avg.val(), accuracy, norm_ED, pred_top_str, pred_mid_str, pred_bot_str, label_top, label_mid, label_bot, infer_time, length_of_data


####################################################
###############kaniblu/hangul-utils#################
####################################################

# encoding: UTF-8
__all__ = ["split_syllable_char", "split_syllables",
           "join_jamos", "join_jamos_char",
           "CHAR_INITIALS", "CHAR_MEDIALS", "CHAR_FINALS"]

import itertools

INITIAL = 0x001
MEDIAL = 0x010
FINAL = 0x100
CHAR_LISTS = {
    INITIAL: list(map(chr, [
        0x3131, 0x3132, 0x3134, 0x3137, 0x3138, 0x3139,
        0x3141, 0x3142, 0x3143, 0x3145, 0x3146, 0x3147,
        0x3148, 0x3149, 0x314a, 0x314b, 0x314c, 0x314d,
        0x314e
    ])),
    MEDIAL: list(map(chr, [
        0x314f, 0x3150, 0x3151, 0x3152, 0x3153, 0x3154,
        0x3155, 0x3156, 0x3157, 0x3158, 0x3159, 0x315a,
        0x315b, 0x315c, 0x315d, 0x315e, 0x315f, 0x3160,
        0x3161, 0x3162, 0x3163
    ])),
    FINAL: list(map(chr, [
        0x3131, 0x3132, 0x3133, 0x3134, 0x3135, 0x3136,
        0x3137, 0x3139, 0x313a, 0x313b, 0x313c, 0x313d,
        0x313e, 0x313f, 0x3140, 0x3141, 0x3142, 0x3144,
        0x3145, 0x3146, 0x3147, 0x3148, 0x314a, 0x314b,
        0x314c, 0x314d, 0x314e
    ]))
}
CHAR_INITIALS = CHAR_LISTS[INITIAL]
CHAR_MEDIALS = CHAR_LISTS[MEDIAL]
CHAR_FINALS = CHAR_LISTS[FINAL]
CHAR_SETS = {k: set(v) for k, v in CHAR_LISTS.items()}
CHARSET = set(itertools.chain(*CHAR_SETS.values()))
CHAR_INDICES = {k: {c: i for i, c in enumerate(v)}
                for k, v in CHAR_LISTS.items()}


def is_hangul_syllable(c):
    return 0xac00 <= ord(c) <= 0xd7a3  # Hangul Syllables


def is_hangul_jamo(c):
    return 0x1100 <= ord(c) <= 0x11ff  # Hangul Jamo


def is_hangul_compat_jamo(c):
    return 0x3130 <= ord(c) <= 0x318f  # Hangul Compatibility Jamo


def is_hangul_jamo_exta(c):
    return 0xa960 <= ord(c) <= 0xa97f  # Hangul Jamo Extended-A


def is_hangul_jamo_extb(c):
    return 0xd7b0 <= ord(c) <= 0xd7ff  # Hangul Jamo Extended-B


def is_hangul(c):
    return (is_hangul_syllable(c) or
            is_hangul_jamo(c) or
            is_hangul_compat_jamo(c) or
            is_hangul_jamo_exta(c) or
            is_hangul_jamo_extb(c))


def is_supported_hangul(c):
    return is_hangul_syllable(c) or is_hangul_compat_jamo(c)


def check_hangul(c, jamo_only=False):
    if not ((jamo_only or is_hangul_compat_jamo(c)) or is_supported_hangul(c)):
        raise ValueError(f"'{c}' is not a supported hangul character. "
                         f"'Hangul Syllables' (0xac00 ~ 0xd7a3) and "
                         f"'Hangul Compatibility Jamos' (0x3130 ~ 0x318f) are "
                         f"supported at the moment.")


def get_jamo_type(c):
    check_hangul(c)
    assert is_hangul_compat_jamo(c), f"not a jamo: {ord(c):x}"
    return sum(t for t, s in CHAR_SETS.items() if c in s)


def split_syllable_char(c):
    """
    Splits a given korean syllable into its components. Each component is
    represented by Unicode in 'Hangul Compatibility Jamo' range.
    Arguments:
        c: A Korean character.
    Returns:
        A triple (initial, medial, final) of Hangul Compatibility Jamos.
        If no jamo corresponds to a position, `None` is returned there.
    Example:
        >>> split_syllable_char("안")
        ("ㅇ", "ㅏ", "ㄴ")
        >>> split_syllable_char("고")
        ("ㄱ", "ㅗ", None)
        >>> split_syllable_char("ㅗ")
        (None, "ㅗ", None)
        >>> split_syllable_char("ㅇ")
        ("ㅇ", None, None)
    """
    check_hangul(c)
    if len(c) != 1:
        raise ValueError("Input string must have exactly one character.")

    init, med, final = None, None, None
    if is_hangul_syllable(c):
        offset = ord(c) - 0xac00
        x = (offset - offset % 28) // 28
        init, med, final = x // 21, x % 21, offset % 28
        if not final:
            final = None
        else:
            final -= 1
    else:
        pos = get_jamo_type(c)
        if pos & INITIAL == INITIAL:
            pos = INITIAL
        elif pos & MEDIAL == MEDIAL:
            pos = MEDIAL
        elif pos & FINAL == FINAL:
            pos = FINAL
        idx = CHAR_INDICES[pos][c]
        if pos == INITIAL:
            init = idx
        elif pos == MEDIAL:
            med = idx
        else:
            final = idx
    return tuple(CHAR_LISTS[pos][idx] if idx is not None else None
                 for pos, idx in
                 zip([INITIAL, MEDIAL, FINAL], [init, med, final]))


def split_syllables(s, ignore_err=True, pad=None):
    """
    Performs syllable-split on a string.
    Arguments:
        s (str): A string (possibly mixed with non-Hangul characters).
        ignore_err (bool): If set False, it ensures that all characters in
            the string are Hangul-splittable and throws a ValueError otherwise.
            (default: True)
        pad (str): Pad empty jamo positions (initial, medial, or final) with
            `pad` character. This is useful for cases where fixed-length
            strings are needed. (default: None)
    Returns:
        Hangul-split string
    Example:
        >>> split_syllables("안녕하세요")
        "ㅇㅏㄴㄴㅕㅇㅎㅏㅅㅔㅇㅛ"
        >>> split_syllables("안녕하세요~~", ignore_err=False)
        ValueError: encountered an unsupported character: ~ (0x7e)
        >>> split_syllables("안녕하세요ㅛ", pad="x")
        'ㅇㅏㄴㄴㅕㅇㅎㅏxㅅㅔxㅇㅛxxㅛx'
    """

    def try_split(c):
        try:
            return split_syllable_char(c)
        except ValueError:
            if ignore_err:
                return (c,)
            raise ValueError(f"encountered an unsupported character: "
                             f"{c} (0x{ord(c):x})")

    s = map(try_split, s)
    if pad is not None:
        tuples = map(lambda x: tuple(pad if y is None else y for y in x), s)
    else:
        tuples = map(lambda x: filter(None, x), s)
    return "".join(itertools.chain(*tuples))


def join_jamos_char(init, med, final=None):
    """
    Combines jamos into a single syllable.
    Arguments:
        init (str): Initial jao.
        med (str): Medial jamo.
        final (str): Final jamo. If not supplied, the final syllable is made
            without the final. (default: None)
    Returns:
        A Korean syllable.
    """
    chars = (init, med, final)
    for c in filter(None, chars):
        check_hangul(c, jamo_only=True)

    idx = tuple(CHAR_INDICES[pos][c] if c is not None else c
                for pos, c in zip((INITIAL, MEDIAL, FINAL), chars))
    init_idx, med_idx, final_idx = idx
    # final index must be shifted once as
    # final index with 0 points to syllables without final
    final_idx = 0 if final_idx is None else final_idx + 1
    return chr(0xac00 + 28 * 21 * init_idx + 28 * med_idx + final_idx)


def join_jamos(s, ignore_err=True):
    """
    Combines a sequence of jamos to produce a sequence of syllables.
    Arguments:
        s (str): A string (possible mixed with non-jamo characters).
        ignore_err (bool): If set False, it will ensure that all characters
            will be consumed for the making of syllables. It will throw a
            ValueError when it fails to do so. (default: True)
    Returns:
        A string
    Example:
        >>> join_jamos("ㅇㅏㄴㄴㅕㅇㅎㅏㅅㅔㅇㅛ")
        "안녕하세요"
        >>> join_jamos("ㅇㅏㄴㄴㄴㅕㅇㅎㅏㅅㅔㅇㅛ")
        "안ㄴ녕하세요"
        >>> join_jamos()
    """
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
