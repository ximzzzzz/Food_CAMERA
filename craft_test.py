#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import os
import time
import argparse
# sys.path.append('/home/Data/FoodDetection/Serving/ocr/pipeline/CRAFT_pytorch')
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from PIL import Image
import cv2
from skimage import io
import numpy as np
import craft_utils
# import test
from test_ import copyStateDict
# from test import test_net
import imgproc
import file_utils
import json
import zipfile
import pandas as pd
import traceback

from craft import CRAFT
from collections import OrderedDict
from crop_words_ import crop, generate_words
from soynlp.hangle import levenshtein, jamo_levenshtein ##20201026 newly added (install through pip if needed)
from craft import CRAFT

from collections import OrderedDict
import matplotlib.pyplot as plt
import matplotlib.image as plt_image
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

import sys
import os
import time, errno
import cv2
from skimage import io
import numpy as np
import pandas as pd
import imgproc as imgproc
import test_
from crop_words_ import crop, generate_words
import torch
import traceback
import crop_words_
import re
sys.path.append('/home/Data/FoodDetection/Serving/ocr/pipeline/CRAFT_pytorch')
from edit_distance import lexicon_search, get_weight_df, fine_filtering, get_weight_ratio, get_image_center, filterNweight
from collections import Counter, defaultdict
home = '/home'


# In[2]:


def get_image_center(pts):
    height_center = (pts[:,0].max() + pts[:,0].min())/2
    width_center = (pts[:,1].max() + pts[:,1].min())/2
    return (int(height_center), int(width_center))


# In[3]:


def get_pts_center(pts):
    height_center = (pts[:,0].max() + pts[:,0].min())/2
    width_center = (pts[:,1].max() + pts[:,1].min())/2
    return (int(width_center), int(height_center))


# In[4]:


def euclidean_distance(x, y):   
#     return np.sqrt(np.sum((x - y) ** 2))
    x = np.asarray(x)
    y = np.asarray(y)
    return np.linalg.norm(x - y)


# In[5]:


def test_net(net, image, text_threshold, link_threshold, low_text, cuda, poly, refine_net=None):
    t0 = time.time()

    # resize
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, args['canvas_size'], interpolation=cv2.INTER_LINEAR, 
                                                                          mag_ratio=args['mag_ratio'])
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
    if cuda:
        x = x.cuda()

    # forward pass
    with torch.no_grad():
        y, feature = net(x)

    # make score and link map
    score_text = y[0,:,:,0].cpu().data.numpy()
    score_link = y[0,:,:,1].cpu().data.numpy()

    # refine link
    if refine_net is not None:
        with torch.no_grad():
            y_refiner = refine_net(y, feature)
        score_link = y_refiner[0,:,:,0].cpu().data.numpy()

    t0 = time.time() - t0
    t1 = time.time()

    # Post-processing
    boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)

    # coordinate adjustment
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None: polys[k] = boxes[k]

    t1 = time.time() - t1

    # render results (optional)
    render_img = score_text.copy()
    render_img = np.hstack((render_img, score_link))
    ret_score_text = imgproc.cvt2HeatmapImg(render_img)

#     if args.show_time : print("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))

    return boxes, polys, ret_score_text

def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict


# In[6]:


#craft_mlt_25k.pth
args = {"trained_model":home+'/Data/FoodDetection/Serving/ocr/pipeline/craft_mlt_25k.pth',
        "text_threshold":0.7,
        "low_text":0.4,
        "link_threshold":0.4,
        "cuda":False,
        "canvas_size":1280,
        "mag_ratio": 1.5,
        "poly":False,
        "show_time":False,
        "test_folder": "/Data//FoodDetection/AI_OCR/CRAFT/",
        "filepath": '/Data//FoodDetection/data/text_detection/RDProject/ocr_1000055.jpg',
        "refine" : False,
         "refiner_model": 'weights/craft_refiner_CTW1500.pth',

        "IMG_WIDTH": 136,
        "IMG_HEIGHT": 136,
        "IMG_CHANNELS": 3,
        "classfication Model": "mobilenet_model"
}

#args = parser.parse_args()

""" For test images in a folder """

filepath = args['filepath']


# In[7]:


image_list = [filepath]
image_names = []
image_paths = []

# CUSTOMISE START
start = args["test_folder"] + '01_src'  # '/Data/CRAFT_process/test_1/01_images'

for num in range(len(image_list)):
    image_names.append(args['filepath'])

crop_path = args["test_folder"] + '03_crop'
data = pd.DataFrame(columns=['image_name', 'word_bboxes', 'pred_words', 'align_text'])
data['image_name'] = image_names

# load net
net = CRAFT()  # initialize
print('Loading weights from checkpoint (' + args["trained_model"] + ')')
if args["cuda"]:
    net.load_state_dict(copyStateDict(torch.load(args["trained_model"])))
else:
    net.load_state_dict(copyStateDict(torch.load(args["trained_model"], map_location='cpu')))

if args["cuda"]:
    net = net.cuda()
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = False

_ = net.eval()


# In[8]:


# LinkRefiner
refine_net = None
if args["refine"]:
    from refinenet import RefineNet

    refine_net = RefineNet()
    if args["cuda"]:
        refine_net.load_state_dict(copyStateDict(torch.load(args["refiner_model"])))
        refine_net = refine_net.cuda()
        refine_net = torch.nn.DataParallel(refine_net)
    else:
        refine_net.load_state_dict(copyStateDict(torch.load(args["refiner_model"], map_location='cpu')))

    refine_net.eval()
    args['poly'] = True

# load data
for k, image_path in enumerate(image_list):
    print("Test image {:d}/{:d}: {:s}".format(k + 1, len(image_list), image_path), end='\r')
    image = imgproc.loadImage(image_path)

    bboxes, polys, det_scores = test_net(net, image, args["text_threshold"],
                                                              args["link_threshold"],
                                                              args["low_text"], args["cuda"], args["poly"],
                                                              refine_net)
    bbox_score = {}

    for box_num in range(len(bboxes)):
        key = str(det_scores[box_num])
        item = bboxes[box_num]
        bbox_score[key] = item

    data['word_bboxes'][k] = bbox_score
    # save score text
    ##filename, file_ext = os.path.splitext(os.path.basename(image_path))
    ##mask_file = result_folder + "/res_" + filename + '_mask.jpg'        # '/Data/CRAFT_process/test_1/02_map/res_' + filename + '_mask.jpg'

    ##print(mask_file)
    ##cv2.imwrite(mask_file, score_text)

    ##file_utils.saveResult(image_path, image[:,:,::-1], polys, dirname=result_folder+'/')
# data.to_csv(args["test_folder"] + 'data.csv', sep=',', na_rep='Unknown')
# del data
# data = pd.read_csv(args["test_folder"] + 'data.csv')
# Crop

for image_num in range(data.shape[0]):
    image = cv2.imread(os.path.join(start, data['image_name'][image_num]))
    image_name = data['image_name'][image_num].strip('.jpg')
    score_bbox = data['word_bboxes'][image_num].split('),')
    cropped_imgs, pts_list, image = generate_words(image_name, score_bbox, image, crop_path)
#                     print('cropped imgs : ',cropped  
    # print(cropped_imgs.shape)

#                 recognition_list = run(cropped_imgs)
#                 if len(recognition_list) == 0 :
#                     print("imagelen: " , len(recognition_list))
#                 print("imagename: " , recognition_list)
#             print('cropped_imgs : ', cropped_imgs)


# In[ ]:


bbox_score = {}

for box_num in range(len(bboxes)):
    key = str(det_scores[box_num])
    item = bboxes[box_num]
    bbox_score[key] = item

data['word_bboxes'][k] = bbox_score

for image_num in range(data.shape[0]):
    image = cv2.imread(os.path.join(start, data['image_name'][image_num]))
    image_name = data['image_name'][image_num].strip('.jpg')
    score_bbox = data['word_bboxes'][image_num].split('),')
    cropped_imgs, pts_list, image = generate_words(image_name, score_bbox, image, crop_path)


# In[ ]:


for image_num in range(data.shape[0]):
    image = cv2.imread(os.path.join(start, data['image_name'][image_num]))
    image_name = data['image_name'][image_num].strip('.jpg')
    score_bbox = data['word_bboxes'][image_num].split('),')
    cropped_imgs, pts_list, image = generate_words(image_name, score_bbox, image, crop_path)


# In[9]:


cropped_list = []
pts_list = []
num_bboxes = len(score_bbox)
for num in range(num_bboxes):
    bbox_coords = score_bbox[num].split(':')[-1].split(',\n')
    if bbox_coords!=['{}']:
      l_t = float(bbox_coords[0].strip(' array([').strip(']').split(',')[0])
      t_l = float(bbox_coords[0].strip(' array([').strip(']').split(',')[1])
      r_t = float(bbox_coords[1].strip(' [').strip(']').split(',')[0])
      t_r = float(bbox_coords[1].strip(' [').strip(']').split(',')[1])
      r_b = float(bbox_coords[2].strip(' [').strip(']').split(',')[0])
      b_r = float(bbox_coords[2].strip(' [').strip(']').split(',')[1])
      l_b = float(bbox_coords[3].strip(' [').strip(']').split(',')[0])
      b_l = float(bbox_coords[3].strip(' [').strip(']').split(',')[1].strip(']'))
      pts = np.array([[int(l_t), int(t_l)], [int(r_t) ,int(t_r)], [int(r_b) , int(b_r)], [int(l_b), int(b_l)]])


# -----

# In[ ]:


import json
import pandas as pd
import os
import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
from torch.utils.data import *
import numpy as np
import time
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import easydict
import cv2

import sys

sys.path.append(home+'/Data/FoodDetection/AI_OCR/Whatiswrong')
sys.path.append(home+'/Data/FoodDetection/AI_OCR//Scatter')
sys.path.append(home+'/Data/FoodDetection/AI_OCR//RobustScanner')
sys.path.append(home+'/Data/FoodDetection/AI_OCR//Nchar_clf')
sys.path.append(home+'/Data/FoodDetection/AI_OCR//EFIFSTR_torch')
import re
import six
import math
import torchvision.transforms as transforms
import utils
import augs
import augs2
# import scatter_model_jamo
import www_model_jamo_vertical
import torch.distributed as dist
# import en_dataset
# import ko_dataset
import albumentations
from albumentations import GaussNoise, IAAAdditiveGaussianNoise, Compose, OneOf
from albumentations.pytorch import ToTensor
# from jamo import h2j, j2hcj, j2h

import pickle
from albumentations.core.transforms_interface import ImageOnlyTransform
# from albumentations.augmentations import functional as F
import BaseModel
import Nchar_utils
import Model
import efifstr_utils
from tensorflow.keras.preprocessing.image import array_to_img
import BaseModel_efif


# In[ ]:


import importlib
importlib.reload(crop_words_)


# In[ ]:


# opt
opt = easydict.EasyDict({
    "experiment_name" : f'{utils.SaveDir_maker(base_model = "RobustScanner", base_model_dir = "./models")}',
    "imgH" : 64 ,
    "imgW" :  256,
    'rgb' :True,
    'top_char' : ' !"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZㄱㄴㄷㄹㅁㅂㅅㅇㅈㅊㅋㅌㅍㅎㄲㄸㅃㅆㅉ',
    'middle_char' : ' ㅏㅑㅓㅕㅗㅛㅜㅠㅡㅣㅐㅒㅔㅖㅘㅙㅚㅝㅞㅟㅢ',
    'mid_char' : ' ㅏㅑㅓㅕㅗㅛㅜㅠㅡㅣㅐㅒㅔㅖㅘㅙㅚㅝㅞㅟㅢ',
    'bottom_char' : ' ㄱㄴㄷㄹㅁㅂㅅㅇㅈㅊㅋㅌㅍㅎㄲㄸㅃㅆㅉㄳㄵㄶㄺㄻㄼㄽㄾㄿㅀㅄ',
    'bot_char' : ' ㄱㄴㄷㄹㅁㅂㅅㅇㅈㅊㅋㅌㅍㅎㄲㄸㅃㅆㅉㄳㄵㄶㄺㄻㄼㄽㄾㄿㅀㅄ',
    'batch_max_length' : 23,
    'num_fiducial' : 20,
    'output_channel' : 512,
    'hidden_size' :256,
    'num_epoch' : 100,
    'input_channel' : 3,
    'trans' : True, 'extract' : 'resnet', 'pred' : '', 'hybrid_direction':2, 'position_direction':2,
    })
device = torch.device('cpu') #utils.py 안에 device는 따로 세팅해줘야함
top_converter = utils.AttnLabelConverter(opt.top_char, device)
middle_converter = utils.AttnLabelConverter(opt.middle_char, device)
bottom_converter = utils.AttnLabelConverter(opt.bottom_char, device)
opt.top_n_cls = len(top_converter.character)
opt.mid_n_cls = len(middle_converter.character)
opt.bot_n_cls = len(bottom_converter.character)


# In[ ]:


model = BaseModel.model(opt, device)
model.load_state_dict(torch.load(home+'/Data/FoodDetection/AI_OCR/models/RobustScanner_1223/0/best_accuracy_96.83.pth', map_location='cpu' if device.type=='cpu' else 'cuda')) 
model.to(device)
_ = model.eval()

prev_model = BaseModel.model(opt, device)
prev_model.load_state_dict(torch.load(home+'/Data/FoodDetection/AI_OCR/models/RobustScanner_1221/0/best_accuracy_94.96.pth', map_location='cpu' if device.type=='cpu' else 'cuda')) 
prev_model.to(device)
_ = prev_model.eval()


# In[ ]:


def recog_run(opt, device, model, cropped_imgs=None):
      

    attn_converter_top = utils.AttnLabelConverter(opt.top_char, device)
    attn_converter_mid = utils.AttnLabelConverter(opt.mid_char, device)
    attn_converter_bot = utils.AttnLabelConverter(opt.bot_char, device)
    opt.top_n_cls_attn = attn_converter_top.n_cls
    opt.mid_n_cls_attn = attn_converter_mid.n_cls
    opt.bot_n_cls_attn = attn_converter_bot.n_cls
    
    if len(cropped_imgs) < 1:
        return ['No detection']
    
    recognition_list = []
    ver_list = []
    for cropped_img in cropped_imgs:
        if not 0 in list(cropped_img.shape):
            ver_list.append(cropped_img)
    cropped_imgs = np.array(ver_list)
    
    try:
        assert len(cropped_imgs[0].shape) == 3  # (batchsize, channels, height, width)
        assert (cropped_imgs[0].shape[0] == 3) | (cropped_imgs[0].shape[2] == 3)
    except:
        print("tensor should have 4 dimension including batch size on 0th index. and make sure positioning channel on 1 or 3th index")
        recognition_list.append("No detection")
        return recognition_list
        #raise (AssertionError('tensor should have 4 dimension including batch size on 0th index. and make sure positioning channel on 1 or 3th index'))
    
    batch_size = cropped_imgs.shape[0]
#     print('-'*40)
#     print('input type : ', type(cropped_imgs))
#     print('input shape : ' ,cropped_imgs.shape)
#     print('-'*40)
    stream = utils.CustomDataset_inf(cropped_imgs, resize_shape=(opt.imgH, opt.imgW), transformer=ToTensor())
    loader = DataLoader(stream, batch_size=batch_size, shuffle=False, num_workers=1)
    iterer = iter(loader)
    img, label_top, label_mid, label_bot = next(iterer)
    img = img.to(device)
    text_mid = text_bot = text_top = torch.tensor(
            [[0, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
              0, 0, 0]]).repeat(batch_size, 1)
    
    

    pred_top, pred_mid, pred_bot = model(img, text_top[:, :-1], text_mid[:, :-1], text_bot[:, :-1], is_train=False)

    _, pred_top_index = pred_top.max(2)
    _, pred_mid_index = pred_mid.max(2)
    _, pred_bot_index = pred_bot.max(2)

    decode_top = attn_converter_top.decode(pred_top_index, torch.FloatTensor(batch_size, opt.batch_max_length + 1,
                                                                             opt.top_n_cls_attn + 2))
    decode_mid = attn_converter_mid.decode(pred_mid_index, torch.FloatTensor(batch_size, opt.batch_max_length + 1,
                                                                             opt.mid_n_cls_attn + 2))
    decode_bot = attn_converter_bot.decode(pred_bot_index, torch.FloatTensor(batch_size, opt.batch_max_length + 1,
                                                                             opt.bot_n_cls_attn + 2))

    
    for i in range(batch_size):
        recognition_list.append(utils.str_combine(decode_top[i], decode_mid[i], decode_bot[i]))
   # return ' '.join(recognition_list)
    return recognition_list


# In[ ]:


def Detection(net, urlFilepath):
    try:
        #t = time.time()

        # CRAFT
        cuda_stats = False
        device = torch.device('cpu')
#         device = torch.device('cuda')
        if device.type == 'cpu':
            cuda_stats = False
        else:
            cuda_stats = True

        #"cuda":False, True를 False로 수정 
        args = {"trained_model":'/data/OCR_code/Pipeline/craft_mlt_25k.pth',
                "text_threshold":0.7,
                "low_text":0.4,
                "link_threshold":0.4,
                "cuda":cuda_stats, 
                "canvas_size":1280,
                "mag_ratio": 1.5,
                "poly":False,
                "show_time":False,
                "test_folder": "/data/OCR_dir/",
                "filepath": 'Data//FoodDetection/data/text_detection/RDProject/ocr_1000056.jpg',
                "refine" : False,
                 "refiner_model": 'weights/craft_refiner_CTW1500.pth'
        }

        #date = time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))

        filename = urlFilepath.split("/")[-1]
        
        # 저장 된 이미지 확인
        #filepath = "/Data/CRAFT_process/test_1/01_images/"+str(date)+filename.rstrip()
        
        filepath = urlFilepath

        if os.path.isfile(filepath):
            #print( "Yes. it is a file")

            ##if sys.argv[1] is null:
            # filepath = args["filepath"]

            # image_list = [args.filepath]
            image_list = [filepath]
            image_names = []
            image_paths = []

            # CUSTOMISE START
            ##start = '/Data/CRAFT_process/test_1/01_images'  
            start = filepath.split(filename)[0]    # 파일 경로에 따라 Flexible하게 결정

            for num in range(len(image_list)):
                image_names.append(os.path.relpath(image_list[num], start))

            ###result_folder = args.test_folder+'02_map'
            ###if not os.path.isdir(result_folder):
            ###    os.mkdir(result_folder)

            crop_path = start+'%s_crop'%(filename.split('.')[0])
            
            if not os.path.isdir(crop_path):
                os.mkdir(crop_path)

            data = pd.DataFrame(columns=['image_name', 'word_bboxes', 'pred_words', 'align_text'])
            data['image_name'] = image_names
            
            box_idx = 0
            bbox_dict = {}

            # load data
            for k, image_path in enumerate(image_list):
#                 print("Test image {:d}/{:d}: {:s}".format(k + 1, len(image_list), image_path), end='\r')
                image = imgproc.loadImage(image_path)

                bboxes, polys, score_text, det_scores = test_.test_net(net, image, args["text_threshold"],
                                                                          args["link_threshold"],
                                                                          args["low_text"], args["cuda"], args["poly"],
                                                                          args)  # refinenet = None

                bbox_score = {}
                bbox_list = []

                for box_num in range(len(bboxes)):
                    if det_scores[box_num] < 0.85: # score filtering
                        continue
                    key = str(det_scores[box_num])
                    item = bboxes[box_num]
                    bbox_dict[box_idx] = item.tolist()
                    box_idx += 1
                    bbox_score[key] = item
                
                data['word_bboxes'][k] = bbox_score
                

            csv_file = start+'%s_data.csv'%(filename.split('.')[0]) ### 처리한 이미지 이름_data.csv

            data.to_csv(csv_file, sep=',', na_rep='Unknown')
            del data

            data = pd.read_csv(csv_file)
            # Crop

            for image_num in range(data.shape[0]):
                image = cv2.imread(os.path.join(start, data['image_name'][image_num]))
                image_name = data['image_name'][image_num].strip('.jpg')
                score_bbox = data['word_bboxes'][image_num].split('),')
                cropped_imgs = crop_words_.generate_words(image_name, score_bbox, image, crop_path, bbox_show=False)
            
            print("=========Text Detection and Crop Ends ============")
              
#         else:
#             raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), filepath)


    except Exception as e:  # 모든 예외의 에러 메시지를 출력할 때는 Exception을 사용
#         print('예외가 발생했습니다.', e)
        traceback.print_exc()
        return str(e), 400
    return [bbox_dict, cropped_imgs], 200
#     return data


# In[ ]:


def get_weight_df(pts_list, img_center, recognition_list):
    euclidean_list = []
    center_point_list = []
    area_list = []
    for pts_ in pts_list:
        center_point = get_pts_center(pts_)
        image[center_point[0]-3 : center_point[0]+3, center_point[1]-3: center_point[1]+3] = (0,255,0)
        center_point_list.append(center_point)
        euclidean = euclidean_distance(img_center, center_point)
        euclidean_list.append(euclidean)
        area = cv2.contourArea(pts_)
        area_list.append(area)
    image_area = image.shape[0] * image.shape[1]
    crop_df = pd.DataFrame({'center_point' : center_point_list, 'euclidean_list' : euclidean_list, 'area_list' : area_list, 
                            'recognition_list' : recognition_list})
    
    return crop_df, image_area


# In[ ]:


def fine_filtering(crop_df, cropped_array, n_crop=4):
    crop_df = crop_df.sort_values(by=['area_list'], ascending=False).head(n_crop)
    filtered_idx = crop_df.index
    cropped_array = cropped_array[filtered_idx]
    
    return crop_df, cropped_array


# In[ ]:


def get_weight_ratio(crop_df, img_center):
    max_dist = euclidean_distance(img_center,(0,0))
    ed_ratio = 1 - (crop_df['euclidean_list'] / max_dist)
    
    area_ratio = [x/image_area for x in crop_df['area_list']]
    if len(area_ratio)==1:
        area_ratio=1.0
    else:
        area_ratio = (np.asarray(area_ratio) - min(area_ratio))/(max(area_ratio) - min(area_ratio)) + 0.2 
    area_ratio = np.log10(area_ratio * 10)
    
    total_ratio = ed_ratio * area_ratio
    return total_ratio.values


# In[ ]:


def filterNweight2(pts_list, img_center, image, cropped_array,  n_crop=4):
    
    # get_weight_df 
    euclidean_list = []
    center_point_list = []
    area_list = []
    for pts_ in pts_list:
        center_point = get_pts_center(pts_)
        center_point_list.append(center_point)
        euclidean = euclidean_distance(img_center, center_point)
        euclidean_list.append(euclidean)
        area = cv2.contourArea(pts_)
        area_list.append(area)
    image_area = image.shape[0] * image.shape[1]
    crop_df = pd.DataFrame({'center_point' : center_point_list, 'euclidean_list' : euclidean_list, 'area_list' : area_list})
    
    # fine_filtering
    crop_df = crop_df.sort_values(by=['area_list'], ascending=False).head(n_crop)
    filtered_idx = crop_df.index
    cropped_array = cropped_array[filtered_idx]
    
    if len(cropped_array)==0:
        
        return [], [] ,[]
        
#     print(f'number of cropped imags : {len(cropped_array)}')
    # get_weight_ratio
    max_dist = euclidean_distance(img_center,(0,0))
    ed_ratio = 1 - (crop_df['euclidean_list'] / max_dist)
    
    area_ratio = [x/image_area for x in crop_df['area_list']]
    if len(area_ratio)==1:
        area_ratio=1.0
    else:
        area_ratio = (np.asarray(area_ratio) - min(area_ratio))/(max(area_ratio) - min(area_ratio)) + 0.2 
    area_ratio = np.log10(area_ratio * 10)
    
    total_ratio = ed_ratio * area_ratio
    
    return cropped_array, total_ratio.values, pts_list[filtered_idx]


# In[18]:


base_dir = home+'/Data/FoodDetection/AI_OCR/CRAFT/test_img'
file_names = os.listdir(base_dir)
abs_paths = [os.path.join(base_dir, x) for x in file_names ]


# In[19]:


import glob


# In[9]:


def get_jpg_files(input_dir):
    file_path_list = []
    for i, (root, dirs, files) in enumerate(os.walk(input_dir)):
#         print(f'root : {root}')
        if len(files)!=0:
            pcm_files_list = glob.glob(root+'/*.jpg')
            file_path_list.extend(pcm_files_list)
    return file_path_list


# In[10]:


def rotate_plate_img(image, pts, cropped_img):
    l_t, r_t, r_b, l_b = pts
    
    if abs(l_t[0] - l_b[0]) < 5:
        return cropped_img
    else:
        if l_t[0] > l_b[0]:
            direction = 'non-clockwise'
        else:
            direction = 'clockwise'

        height = l_t[0] - l_b[0]
        hypotenus = np.linalg.norm(l_t - l_b)

        if height == hypotenus:
            return cropped_img

        else:
            if direction =='clockwise':
                angle = np.degrees(np.arcsin(height / hypotenus))
#                 side_angle = 180-temp_angle
#                 angle = 180 - (side_angle + 90)
            else:
                angle = np.degrees(np.arcsin(height / hypotenus))
        #     if height > width :
        #         angle = -angle

            center_coord = get_image_center(np.array([[0,0], [image.shape[0]-1, image.shape[1]-1]]))
            rotation_matrix = cv2.getRotationMatrix2D(center =center_coord, angle=angle, scale=1 )
            rotated_img = cv2.warpAffine(image, M=rotation_matrix, dsize= (image.shape[1], image.shape[0]))
            
            pts_refine = np.column_stack([pts, np.ones((4,1))])
            rotation_matrix_refine = np.concatenate([rotation_matrix, np.array([[0,0,1]])], axis=0)
            pts_rotated = np.dot(rotation_matrix_refine, pts_refine.transpose())
            
            y_min = int(pts_rotated[1].min())
            y_max = int(pts_rotated[1].max())
            x_min = int(pts_rotated[0].min())
            x_max = int(pts_rotated[0].max())
            y_expand_length = int((y_max - y_min)*0.05)
            x_expand_length = int((x_max - x_min)*0.05)
            cropped_img = rotated_img[y_min - y_expand_length : y_max + y_expand_length , x_min - x_expand_length : x_max + x_expand_length]
    
        return cropped_img


# In[22]:


path_list = get_jpg_files(home + '/Data/FoodDetection/data/lexicon/raw/')


# In[23]:


# iterer = iter(path_list)


# In[989]:


# path = next(iterer)
random_path = np.random.choice(path_list, size=1)[0]
[bbox_dict, (cropped_array, pts_list, image)], res_code = Detection(net, random_path)


# In[990]:


img_center = get_image_center(np.array([[0,0], [image.shape[0]-1, image.shape[1]-1]])) 
cropped_array, total_ratio, pts = filterNweight2(pts_list, img_center, image, cropped_array, n_crop=4)

plate_array = []
for cropped_img, pt in zip(cropped_array, pts):
    plate_array.append(rotate_plate_img(image, pt, cropped_img))
plate_array = np.array(plate_array)


# In[991]:


prev_crop = recog_run(opt, 'cpu', prev_model, cropped_array)
prev_plate = recog_run(opt, 'cpu', prev_model, plate_array)
recognition_list = recog_run(opt, 'cpu', model, plate_array)
pd.DataFrame({'prev_crop' : prev_crop, 'prev_plate' : prev_plate, 'new_plate' : recognition_list})


# In[971]:


print(prev_crop,'\n',prev_plate,'\n', recognition_list)


# In[985]:


image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# image[ img_center[0]-5 : img_center[0]+5, img_center[1]-5: img_center[1]+5] = (0,0, 255)
# image[ pts_list[0][0][1]-10 : pts_list[0][0][1]+10, pts_list[0][0][0]-10: pts_list[0][0][0]+10] = (255, 0, 0 )
Image.fromarray(image)


# In[ ]:


# crop_df, image_area = get_weight_df(pts_list, img_center, recognition_list)


# In[81]:


# crop_df, cropped_array = fine_filtering(crop_df, cropped_array )


# In[973]:


Image.fromarray(cv2.cvtColor(plate_array[0], cv2.COLOR_BGR2RGB))


# In[974]:


Image.fromarray(cv2.cvtColor(cropped_array[0], cv2.COLOR_BGR2RGB))


# In[36]:


# total_ratio = get_weight_ratio(crop_df, img_center)
# total_ratio


# ## lexicon search

# In[1003]:


example = '포도100_Taslioog_자연은_WoongiIn'
example = example.split('_')
# example = recognition_list


# -------

# In[1011]:


import kenlm


# In[1035]:


# model = kenlm.Model('/home/Data/etc/Robust_ASR/data/zeroth/zeroth.lm.fg.arpa')
import pickle


# In[1038]:


small_lm = pickle.load(open('/home/Data/etc/Robust_ASR/data/zeroth/zeroth.lm.tgmed.arpa', 'rb'))


# In[1039]:


small_lm = kenlm.LanguageModel('/home/Data/etc/Robust_ASR/data/zeroth/zeroth.lm.tgmed.arpa')


# In[1073]:


sum(prob for prob, _,_ in small_lm.full_scores(''))


# In[1081]:


small_lm.score('thi')


# In[17]:


get_ipython().system(' pip install sentencepiece')


# In[22]:


sys.path.append('/home/Data/FoodDetection/AI_OCR/embedding')
from models.word_eval import WordEmbeddingEvaluator


# In[ ]:


WordEmbeddingEvaluator(vecs_tex)


# ----------

# In[ ]:


### 고려사항
- 제조사만 동일하고 제품명이 다를 경우 거르기 -> threshold 기준을 어떻게 설정할 것인가(main 키워드와 운이좋게 매치됬을경우 고려)
- 제조사 영어로 되있을시 


# In[43]:


lexicon = pd.read_csv(home+'/Data/FoodDetection/Serving/ocr/pipeline/OCR_lexicon_pre.csv' ,error_bad_lines=False)


# In[1007]:


lexicon[lexicon['preprocess'].apply(lambda x : True if re.compile('알로에').search(x) else False)]


# In[276]:


# lexicon = pd.concat([lexicon, pd.DataFrame([{'FOOD_CD' : 'G012421445', 'preprocess' : 'lotte_행복한주말_사랑나눔'}]) ], axis=0 , ignore_index=True) 


# In[201]:


example = recognition_list


# In[1004]:


rest_sum = sum(list(filter(lambda x : x >= total_ratio.mean(), total_ratio)))
threshold =  (rest_sum * 0.7 )/ (len(example)+2)
# if rest_sum < 1 else  rest_sum /  (len(example) * 10**-1)
threshold


# In[1005]:


start_time = time.time()
ed_list = []
eps = 10e-4
dynamic_dict = defaultdict(Counter)
for word in lexicon['preprocess']:
    min_ed = []
    word_split = word.split('_')
    for idx, each in enumerate(example):
        each_ed = []
        for word_ in word_split:
            if dynamic_dict[idx][word_] !=0:  
                er = dynamic_dict[idx][word_]
                each_ed.append(er)
            else:
                ed = levenshtein(each.lower(), word_.lower())
                er = 1- (ed/(max(len(word_), len(each))))
                if er <= 0.4:
                    er = 0 + eps
#             print(f'"{each.lower()}"와 "{word.lower()}"속 단어 "{word_.lower()}" 의  raw ED 값 : {ed},  1- levenshtein 값  : {er}')
#             print(f'"{each.lower()}"와 "{word_.lower()}" 의  raw ED 값 : {ed},  1- levenshtein 값  : {er}')
                each_ed.append(er)
                dynamic_dict[idx][word_] = er
                ### 2021-01-28 추가 ##
                if er == 1.0:
                    break
#         min_ed.append(ed_ratio[idx] * max(each_ed))
        min_ed.append(total_ratio[idx] * max(each_ed))
#         min_ed.append(min(each_ed))
    ed_list.append(sum(min_ed)/len(example))

lexicon['ed'] = ed_list

max_data = lexicon[lexicon['ed'] == lexicon['ed'].max()]
if max_data['ed'].max() < threshold:
    print(f'not in lexicon(ED : {max_data["ed"].max()}, threshold : {threshold})')
else:
    most_similar_idx = max_data['preprocess'].apply(lambda x :  abs(len(example) - len(x.split('_')))).sort_values().index[0]
    print(f'Search Result : {lexicon.loc[most_similar_idx]["preprocess"]}, FOOD CD : {lexicon.loc[most_similar_idx]["FOOD_CD"]},  ED : {round(lexicon.loc[most_similar_idx]["ed"],2)}')
print('elapsed time : ', time.time() - start_time)


# In[1006]:


lexicon = lexicon.sort_values(by = ['ed'],ascending=False)
lexicon.head(30)


# In[32]:


######### old one
start_time = time.time()
ed_list = []

for word in lexicon['preprocess']:
    min_ed = []
    word_split = word.split('_')
    for idx, each in enumerate(example):
        each_ed = []
        for word_ in word_split:
            ed = levenshtein(each.lower(), word_.lower())
            er = 1- (ed/(max(len(word_), len(each))))
            if er <= 0.4:
                er = 0
#             print(f'"{each.lower()}"와 "{word.lower()}"속 단어 "{word_.lower()}" 의  raw ED 값 : {ed},  1- levenshtein 값  : {er}')
#             print(f'"{each.lower()}"와 "{word_.lower()}" 의  raw ED 값 : {ed},  1- levenshtein 값  : {er}')
            each_ed.append(er)
#         min_ed.append(ed_ratio[idx] * max(each_ed))
        min_ed.append(total_ratio[idx] * max(each_ed))
#         min_ed.append(min(each_ed))
    ed_list.append(sum(min_ed)/len(example))

lexicon['ed'] = ed_list

max_data = lexicon[lexicon['ed'] == lexicon['ed'].max()]
if max_data['ed'].max() < threshold:
    print(f'not in lexicon(ED : {max_data["ed"].max()}, threshold : {threshold})')
else:
    most_similar_idx = max_data['preprocess'].apply(lambda x :  abs(len(example) - len(x.split('_')))).sort_values().index[0]
    print(f'Search Result : {lexicon.loc[most_similar_idx]["preprocess"]}, FOOD CD : {lexicon.loc[most_similar_idx]["FOOD_CD"]},  ED : {round(lexicon.loc[most_similar_idx]["ed"],2)}')
print('elapsed time : ', time.time() - start_time)


# In[600]:


lexicon = lexicon.loc[[655]]


# In[116]:


lexicon.loc[2301]['preprocess']


# In[ ]:




