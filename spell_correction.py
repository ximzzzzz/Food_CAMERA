#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from soynlp.hangle import levenshtein, jamo_levenshtein
import re


# In[2]:


home = '/home'
# lexicon_path = home + '/Data/FoodDetection/Serving/ocr/pipeline/OCR_lexicon_pre.csv'
lexicon_path = home + '/Data/FoodDetection/AI_OCR/lexicon_update_pre.csv'
lexicon = pd.read_csv(lexicon_path)
# orig_lexicon = lexicon['preprocess'][:2151]


# In[4]:


orig_lexicon = lexicon['preprocess']


# In[4]:


word_count = defaultdict(Counter)
for word in orig_lexicon:
    word_split = word.split('_')
    for word_splitted in word_split:
        word_count[0][word_splitted] +=1


# In[20]:


# under ed 1 words algorithm
def get_ed1_words(test_word):
    ex_words = []
    for i in range(len(test_word)):
        pre_word = test_word[:i]
        post_word = test_word[i+1:]
        excluded_word = pre_word + post_word
        ex_words.append(excluded_word)
    return ex_words + [test_word]


# In[21]:


# ed less than 2 in lexicon word
word_count_ed = defaultdict(Counter)
for word in orig_lexicon:
    word_split = word.split('_')
    for word_splitted in word_split:
        ed1_words = get_ed1_words(word_splitted)
        for ed1_word in ed1_words:
            word_count_ed[0][ed1_word] +=1
        word_count_ed[0][word_splitted]


# In[7]:


word_count[0]


# In[112]:


# test ed search
test_word = '아몬디'
ed_list = {}
for lexicon_word in word_count[0].keys():
    if levenshtein(test_word, lexicon_word)==1:
        print(lexicon_word)
#     ed = jamo_levenshtein(test_word, lexicon_word)
#     ed_list[lexicon_word] = round(ed,3)
    
# ed_df = pd.DataFrame({'food' : list(ed_list.keys()), 'ed' : list(ed_list.values())}, index=range(len(ed_list))).sort_values(by='ed')
# min_ed = ed_df['ed'].values[0]
# if min_ed > 0.667: #한글자만 다른경우 0.667 마지노선
#     print('better to keep rather than correct')
# else:
#     corr = ed_df['food'].values[0]


# In[113]:


test_words = get_ed1_words(test_word)


# In[115]:


wrong_words = []
for test_ in test_words:
    if word_count_ed[0][test_] ==0:
        wrong_words.append(test_)


# In[116]:


wrong_words


# In[117]:


longest_word = sorted(wrong_words, key=lambda x : len(x), reverse=True )[0]


# In[126]:


length = len(wrong_words)
for char in longest_word:
    have_char = [True if x.find(char)!=-1 else False for x in list(wrong_words)]
    if sum(have_char)==length:
        typo = char
print(typo)


# In[127]:


typo_idx = test_word.index(typo)
pre_word = test_word[:typo_idx]
post_word = test_word[typo_idx+1:]
exception_word = pre_word + post_word


# In[135]:


ed_1 = []
for product in word_count[0].keys():
    if levenshtein(exception_word, product) ==1:
        ed_1.append(product)
#         correction_word = product
jamo_ed_1 = {}
for ed_ in ed_1:
    jamo_ed = round(jamo_levenshtein(exception_word, ed_),2)
    jamo_ed_1[ed_] = jamo_ed


# In[136]:


ed_1


# In[137]:


jamo_ed_1


# ---------

# ## 그냥 해보자

# In[53]:


import time
from multiprocessing import Pool
import multiprocessing as mp
from functools import partial
from itertools import repeat


# In[51]:


recognition_list = ['Minj', '오에스', '흐태', '버더와플']


# In[21]:


def get_ed_1(test_word,word_count_split):
    ed_1 = []
    for lexicon_word in word_count_split:
        if levenshtein(test_word, lexicon_word)==1:
            ed_1.append(lexicon_word)
    return ed_1


# In[81]:


start_time = time.time()
num_cores = mp.cpu_count()
num_cores_use = int(num_cores/4)
word_count_split = np.array_split(list(word_count[0].keys()), num_cores_use)

corrected_idx = np.ones(len(recognition_list))
with Pool(num_cores_use) as pool:
    for idx, test_word in enumerate(recognition_list):

        ed_1 = list(filter(lambda x: len(x)!=0, pool.starmap(get_ed_1, zip(repeat(test_word), word_count_split))))
        ed_1 = [two_dimensions for one_dimension in ed_1 for two_dimensions in one_dimension]
        if len(ed_1)==0:
#             print(f'test_word : {test_word} -> {test_word}')
            continue
        
        jamo_ed_1 = {}
        for ed_ in ed_1:
            jamo_ed = round(jamo_levenshtein(test_word, ed_),4)
            jamo_ed_1[ed_] = jamo_ed
        if len(jamo_ed_1)==0:
#             print(f'test_word : {test_word} -> {test_word}')
            continue
#         print(f'test_word : {test_word} -> {sorted(jamo_ed_1.items(), key=lambda x : x[1])[0][0]}')
        recognition_list[idx] = sorted(jamo_ed_1.items(), key=lambda x : x[1])[0][0]
        corrected_idx[idx] = 1.2
print(time.time() - start_time)


# In[77]:


recognition_list


# In[82]:


corrected_idx


# --------------

# In[128]:


def get_correction(test_word):
    words_ed1 = get_ed1_words(test_word)
    test_words = words_ed1 + [test_word]
    
    wrong_words = []
    for test_ in test_words:
        if word_count_ed[0][test_] ==0:
            wrong_words.append(test_)
    print(wrong_words)
    if not wrong_words :
        return test_word
    
    longest_word = sorted(wrong_words, key=lambda x : len(x), reverse=True )[0]
    
    length = len(longest_word)
    for char in longest_word:
        have_char = [True if x.find(char)!=-1 else False for x in wrong_words]
        if sum(have_char)==length:
            typo = char
            
    typo_idx = test_word.index(typo)
    pre_word = test_word[:typo_idx]
    post_word = test_word[typo_idx+1:]
    exception_word = pre_word + post_word

    for product in word_count[0].keys():
        if levenshtein(exception_word, product) ==1:
            correction_word = product
            return correction_word
    
    return test_word


# In[129]:


get_correction('jawelfj')


# In[121]:


list(filter(lambda x : len(x) < 2, orig_lexicon))


# In[80]:


from tqdm import tqdm


# In[119]:


refined_lexicon = []
for product in tqdm(orig_lexicon):
    refined_product = product + '_'
    
    product_splits = product.split('_') # [매일우유, 저지방&칼슘]
    
    for product_split in product_splits:
        for target in orig_lexicon:
            target_splits = target.split('_') #[매일우유, 저지방&칼슘]
            for target_split in target_splits:
                if target_split=='':
                    continue
#                 print(f'test_split -> {target_split} 차례')
                if product_split==target_split:
                    continue
                elif target_split in product_split:
                    split_again = product_split.replace(target_split, '')
                    
                    if not '_'+split_again in refined_product:
#                         print(f'{product_split} 항목에 현재 상태는 {refined_product}인데 {split_again}이 없으므로 추가하겠습니다!')
                        refined_product = refined_product +split_again + '_'

                    if not '_'+target_split in refined_product:
#                         print(f'{product_split} 항목에 현재 상태는 {refined_product}인데 {target_split}이 없으므로 추가하겠습니다!')
                        refined_product = refined_product +target_split + '_'
                        
    refined_lexicon.append(refined_product.strip('_'))


# In[120]:


refined_lexicon


# In[123]:


def get_factorial(product):
    factorial = []
    splitted_product = product.split('_')
    if len(splitted_product)==1:
        return [product]
    for idx, splitted in enumerate(splitted_product):
        forward_split = splitted_product[:idx]
        backward_split = splitted_product[idx+1: ]
        except_split = forward_split + backward_split
        for split in except_split:
            factorial.append(splitted+'_'+split)
    return factorial


# In[125]:


def get_bigram_lexicon(orig_lexicon):
    bigram_lexicon = []
    for product in orig_lexicon:
        facto = get_factorial(product)
        bigram_lexicon = bigram_lexicon + facto
    return bigram_lexicon


# In[140]:


refined_bigram = get_bigram_lexicon(refined_lexicon)


# In[142]:


refined_bigram = list(set(refined_bigram))
len(refined_bigram)


# In[9]:


bigram_lexicon = list(set(bigram_lexicon))
bigram_lexicon


# In[130]:


from copy import deepcopy 

def bigram_combination(recognition_list, index):
    target = recognition_list[index]
    recog_list = deepcopy(recognition_list)
    recog_list.remove(target)
    combination = []
    for recog in recog_list:
        combi = (target+'_'+recog).strip('_')
        combination.append(combi)
    return combination


# In[12]:


from multiprocessing import Pool
import multiprocessing as mp
from functools import partial
from itertools import repeat
import time


# In[162]:


# pd.DataFrame({'bigram_lexicon' : refined_bigram}).to_csv('bigram_lexicon_refined.csv', index=False)
refined_bigram = pd.read_csv('bigram_lexicon_refined.csv')


# In[176]:


recognition_list = ['우유', 'Q', '11등급', 'MRILIII']
combinations = bigram_combination(recognition_list, 0)
combinations


# In[177]:


start_time = time.time()
num_cores = mp.cpu_count()
# bigram_lexicon_split = np.array_split(list(bigram_lexicon), int(num_cores/2))
bigram_lexicon_split = np.array_split(list(refined_bigram['bigram_lexicon']), int(num_cores/2))
with Pool(int(num_cores/2)) as pool:
    for idx, target in enumerate(recognition_list):
        corrected = False
        edit_distance1 = []
        target_combos = bigram_combination(recognition_list, idx)
        for target_combo in target_combos:
    #         for lexicon_word in bigram_lexicon:
    #             if levenshtein(target_combo, lexicon_word)==1:
    #                 edit_distance1.append(lexicon_word)
            edit_distance1 = list(filter(lambda x : len(x)!=0, pool.starmap(get_editdistance1, zip(repeat(target_combo.lower()), bigram_lexicon_split))))
            edit_distance1 = [two_dimension for one_dimension in edit_distance1 for two_dimension in one_dimension]
            if len(edit_distance1)==0:
#                 print(f'target : {target}, target_combo : {target_combo}, nothing correct')
                recognition_list[idx] = target
                
            elif target_combo in edit_distance1:
                continue
            
            else:
                print(f'target : {target}, target_combo : {target_combo}, edit distance under 2 {edit_distance1}')
                least_ed = 3
                correction_result = target_combo
                # correction_list = {}
                for ed1_lexicon in edit_distance1:
                    jamo_ed = jamo_levenshtein(ed1_lexicon, target_combo)
        #             print(f'jamo levenshtein [{target_} : {ed_nominated}] = {jamo_ed}')
                    if least_ed >= jamo_ed:
                        least_ed = jamo_ed
                        correction_result = ed1_lexicon

                correction_result = correction_result.split('_')[0]
                recognition_list[idx] = correction_result
                corrected = True

            if corrected:
                print(f'correction process finished with {correction_result} stop next search')
                break   

print(time.time() - start_time)


# In[178]:


recognition_list


# In[15]:


num_cores = mp.cpu_count()
bigram_lexicon_split = np.array_split(list(bigram_lexicon), num_cores/2)


# In[70]:


def get_editdistance1(combination, bigram_lexicon):
    ed1_word = []
    for lexicon_word in bigram_lexicon:
        ed = levenshtein(combination, lexicon_word)
        if ed==0:
            return [combination]
        elif ed<=2:
            ed1_word.append(lexicon_word)
    return ed1_word


# In[69]:


def mp_search(bigram_lexicon_split):

    pool = Pool(int(num_cores/2))
#     map_filter_res = list(filter(lambda x : len(x)!=0, pool.map(partial(parallel_func, combination = combinations[0]), bigram_lexicon_split)))
    map_filter_res = list(filter(lambda x : len(x)!=0, pool.starmap(parallel_func, zip(repeat(combinations[0]), bigram_lexicon_split))))
    pool.close()
    pool.join()
    
    return map_filter_res[0]


# In[ ]:


pool.starmap(func, zip(a_args, repeat(second_arg)))


# In[70]:


mp_search(bigram_lexicon_split)


# In[107]:


list(bigram_lexicon).index('오렌지_고칼슘')


# In[ ]:




