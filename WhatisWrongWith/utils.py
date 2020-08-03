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
import re
import pandas as pd

device = torch.device('cuda')
# device = torch.device('cpu')

import numpy as np
import torch
import torch.nn.functional as F


def ClassCounter(data, position, converter):
    if position =='top':
        pos_idx = 1
    elif position =='mid':
        pos_idx = 2
    elif position =='bot':
        pos_idx = 3
    
    cls_stat = []
    for label in np.asarray(data)[:,pos_idx]:
        cls_stat.extend(label)
    
    cls_cnt = np.unique(cls_stat, return_counts=True)
    cls_cnt = pd.DataFrame({'class' : cls_cnt[0], 'count' : cls_cnt[1]})
    max_cnt = cls_cnt['count'].max()
    
    cnt_dict = {}
    for cls in converter.dict.keys():
        n_cls = cls_cnt[cls_cnt['class'] == cls]['count'].values
        if len(n_cls)==0:
            n_cls = [max_cnt]
        cls_enc = converter.dict[cls]
#         print(f'{cls} : {n_cls[0]}')
        cnt_dict[cls_enc] = n_cls[0]
        
    return cnt_dict

def reduced_focal_cbloss(labels, logits, alpha, gamma, threshold):
    
    BCLoss = F.binary_cross_entropy_with_logits(input = logits, target = labels,reduction = "none")
    p = labels*torch.sigmoid(logits)+(1-labels)*(1-torch.sigmoid(logits))
    modulator = torch.where(p < threshold, torch.ones_like(p), ((1-p)/threshold)**gamma)
    loss = modulator * BCLoss

#     weighted_loss = alpha * loss
    focal_loss = torch.sum(loss)

    focal_loss /= torch.sum(labels)
    return focal_loss    


def focal_loss(labels, logits, alpha, gamma):
    """Compute the focal loss between `logits` and the ground truth `labels`.
    Focal loss = -alpha_t * (1-pt)^gamma * log(pt)
    where pt is the probability of being classified to the true class.
    pt = p (if true class), otherwise pt = 1 - p. p = sigmoid(logit).
    Args:
      labels: A float tensor of size [batch, num_classes].
      logits: A float tensor of size [batch, num_classes].
      alpha: A float tensor of size [batch_size]
        specifying per-example weight for balanced cross entropy.
      gamma: A float scalar modulating loss from hard and easy examples.
    Returns:
      focal_loss: A float32 scalar representing normalized total loss.
    """    
    BCLoss = F.binary_cross_entropy_with_logits(input = logits, target = labels,reduction = "none")

    if gamma == 0.0:
        modulator = 1.0
    else:
        modulator = torch.exp(-gamma * labels * logits - gamma * torch.log(1 + torch.exp(-1.0 * logits)))

    loss = modulator * BCLoss

    weighted_loss = alpha * loss
    focal_loss = torch.sum(weighted_loss)

    focal_loss /= torch.sum(labels)
    return focal_loss



def CB_loss(labels, logits, samples_per_cls, no_of_classes, loss_type, beta, gamma, threshold=0.5):
    """Compute the Class Balanced Loss between `logits` and the ground truth `labels`.
    Class Balanced Loss: ((1-beta)/(1-beta^n))*Loss(labels, logits)
    where Loss is one of the standard losses used for Neural Networks.
    Args:
      labels: A int tensor of size [batch].
      logits: A float tensor of size [batch, no_of_classes].
      samples_per_cls: A python list of size [no_of_classes].
      no_of_classes: total number of classes. int
      loss_type: string. One of "sigmoid", "focal", "softmax".
      beta: float. Hyperparameter for Class balanced loss.
      gamma: float. Hyperparameter for Focal loss.
    Returns:
      cb_loss: A float tensor representing class balanced loss
    """
    effective_num = 1.0 - np.power(beta, samples_per_cls)
    weights = (1.0 - beta) / np.array(effective_num)
    weights = weights / np.sum(weights) * no_of_classes

    labels_one_hot = F.one_hot(labels, no_of_classes).float()

    weights = torch.tensor(weights).float().to(device)
    weights = weights.unsqueeze(0)
    weights = weights.repeat(labels_one_hot.shape[0],1) * labels_one_hot
    weights = weights.sum(1)
    weights = weights.unsqueeze(1)
    weights = weights.repeat(1,no_of_classes)

    if loss_type == "focal":
        cb_loss = focal_loss(labels_one_hot, logits, weights, gamma)
    elif loss_type == "sigmoid":
        cb_loss = F.binary_cross_entropy_with_logits(input = logits,target = labels_one_hot, weights = weights)
    elif loss_type == "softmax":
        pred = logits.softmax(dim = 1)
        cb_loss = F.binary_cross_entropy(input = pred, target = labels_one_hot, weight = weights)
    elif loss_type=='reduced_focal':
        cb_loss =reduced_focal_cbloss(labels_one_hot, logits, weights, gamma, threshold)
    return cb_loss



def reduced_focal_loss(pred, target, ignore_index, alpha, gamma, threshold):
    ce_loss = torch.nn.functional.cross_entropy(pred, target, ignore_index=0, reduction='none')
    pt = torch.exp(-ce_loss)
    pt_scaled = ((1-pt)**2)/(0.5**2)
    fr = torch.where(pt < 0.5, torch.ones_like(pt).to(device), pt_scaled)
    return (fr*ce_loss).mean()


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
#         if (os.path.exists(os.path.join(base_model_dir, directory))) and (os.path.isfile(os.path.join(base_model_dir, directory, 'best_accuracy.pth'))):
        if (os.path.exists(os.path.join(base_model_dir, directory))) and (bool(list(filter(re.compile('best_accuracy.*').match, os.listdir(os.path.join(base_model_dir, directory)))))):
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
#         print(f'original Width : {orig_W}, original Height : {orig_H}, ratio : {ratio}')
        if math.ceil(self.resize_H * ratio) > self.resize_W:
#             print(f' expected width : {self.resize_H * ratio}, but replaced with {self.resize_W}')
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
