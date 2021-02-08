#!/usr/bin/env python
# coding: utf-8

# In[37]:


import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from soynlp.hangle import levenshtein, jamo_levenshtein
import re


# In[5]:


home = '/home'
# lexicon_path = home + '/Data/FoodDetection/Serving/ocr/pipeline/OCR_lexicon_pre.csv'
lexicon_path = home + '/Data/FoodDetection/AI_OCR/lexicon_update_pre.csv'
lexicon = pd.read_csv(lexicon_path)
# orig_lexicon = lexicon['preprocess'][:2151]


# In[13]:


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


# In[44]:


# test ed search
test_word = '카리'
ed_list = {}
for lexicon_word in word_count[0].keys():
#     if levenshtein(test_word, lexicon_word)==1:
#         print(lexicon_word)
    ed = jamo_levenshtein(test_word, lexicon_word)
    ed_list[lexicon_word] = round(ed,3)
    
pd.DataFrame({'food' : list(ed_list.keys()), 'ed' : list(ed_list.values())}, index=range(len(ed_list))).sort_values(by='ed')['food'].values[0]


# In[67]:


test_words = get_ed1_words(test_word)


# In[69]:


wrong_words = []
for test_ in test_words:
    if word_count_ed[0][test_] ==0:
        wrong_words.append(test_)


# In[70]:


wrong_words


# In[71]:


longest_word = sorted(wrong_words, key=lambda x : len(x), reverse=True )[0]


# In[74]:


longest_word


# In[73]:


length = len(longest_word)
for char in longest_word:
    have_char = [True if x.find(char)!=-1 else False for x in wrong_words]
    if sum(have_char)==length:
        typo = char
print(typo)


# In[31]:


typo_idx = test_word.index(typo)
pre_word = test_word[:typo_idx]
post_word = test_word[typo_idx+1:]
exception_word = pre_word + post_word


# In[32]:


exception_word


# In[100]:


for product in word_count[0].keys():
    if levenshtein(exception_word, product) ==1:
        correction_word = product


# In[101]:


correction_word


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


# In[ ]:




