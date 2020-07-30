#!/usr/bin/env python
# coding: utf-8

# Code for a data analysis project on presidential speeches and adtempting to predict intelligence scores of the speeches based on word frequency.
# Extracts all the filler words and counts word frequencies.

# @author Stefan Robb

# In[1]:


import nltk
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

wbrd = r"C:\Users\stefa\Desktop\Queens\Fourth Year\CMPE251\Project\ICperSpeech_WordMAtrix.csv"
wb = pd.read_csv(wbrd)
poslist = {}
wordlist = []
txt = ""
for col in wb.columns:
    poslist[col] = []
    wordlist.append(col)
del poslist["IC"]
for i in range(1,3):
    print("Speech" + str(i))
    txtrd = r"C:\Users\stefa\Desktop\Queens\Fourth Year\CMPE251\Project\President Speeches\presidentspeeches\\" + str(i) + ".txt"
    with open(txtrd, "r") as f:
        txt += f.read()
    token = word_tokenize(txt)
    wordtag = nltk.pos_tag(token)
    for posword in wordtag:
        word, pos = posword
        if word in poslist:
            poslist[word].append(pos)
for i in poslist:
    if not poslist[i]:
        poslist[i] = most_frequent(poslist[i])


# In[23]:


print(poslist)


# In[18]:


def most_frequent(List): 
    counter = 0
    num = List[0]
      
    for i in List: 
        curr_frequency = List.count(i)
        if(curr_frequency> counter): 
            counter = curr_frequency 
            num = i 
  
    return num 


# In[13]:


stop_words = set(stopwords.words('english'))
filteredout_words = []
filtered_words = []
for w in wordlist: 
    if w not in stop_words: 
        filtered_words.append(w) 
    else:
        filteredout_words.append(w)
print(filteredout_words)

