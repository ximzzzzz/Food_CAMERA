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


