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


# In[3]:


orig_lexicon = lexicon['preprocess']


# In[39]:


list(filter(lambda x : re.compile('카레').search(x), orig_lexicon))


# In[15]:


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


# In[108]:


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


# In[114]:


test_words


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


# In[188]:


test_word = '농삼'
ed_1 = []
for lexicon_word in word_count[0].keys():
    if levenshtein(test_word, lexicon_word)==1:
        ed_1.append(lexicon_word)

jamo_ed_1 = {}
for ed_ in ed_1:
    jamo_ed = round(jamo_levenshtein(test_word, ed_),4)
    jamo_ed_1[ed_] = jamo_ed


# In[189]:


jamo_ed_1


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


# In[5]:


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


# In[6]:


bigram_lexicon = []
for product in orig_lexicon:
    facto = get_factorial(product)
    bigram_lexicon = bigram_lexicon + facto


# In[9]:


bigram_lexicon = set(bigram_lexicon)


# In[10]:


# test ed search
test_word = ['카리','3분']
test_list_corr = '스팬_클래식'
ed_list = []
for lexicon_word in bigram_lexicon:
    if levenshtein(test_list_corr, lexicon_word)==1:
        ed_list.append(lexicon_word)

least_ed = 1
correction_name = test_list_corr
# correction_list = {}
for ed_nominated in ed_list:
    jamo_ed = jamo_levenshtein(ed_nominated, test_list_corr)
    if least_ed > jamo_ed:
        least_ed = jamo_ed
        correction_name = ed_nominated
#     correction_list[ed_nominated] =jamo_ed
correction_name


# In[221]:


list(bigram_lexicon).index('스팸_클래식')


# In[196]:


list(correction_list)


# In[ ]:




