#!/usr/bin/env python
# coding: utf-8

# In[8]:


import urllib.request
import os
import tarfile
import re
import os


# In[2]:


url="http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
filepath="data/aclImdb_v1.tar.gz"

#如果檔案不存在 就去download .gz 回來吧
if not os.path.isfile(filepath):
    print('start to download.....')
    result=urllib.request.urlretrieve(url,filepath)
    print('downloaded:',result)


# In[3]:


#如果檔案不存在 就從.gz 解壓縮吧
if not os.path.exists("data/aclImdb"):
    tfile = tarfile.open("data/aclImdb_v1.tar.gz", 'r:gz')
    result=tfile.extractall('data/')


# # 1. Import Library

# In[4]:


from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer


# # 資料準備

# In[5]:


#import re
# r代表了原字符串 , # compile(pattern, flags=0)
# 將正則表達式轉換爲一個「pattern object」稱之爲「模式對象」。將其保存下來，備後續用。
# 評論文字內 會有很多html的 tag, 若符合 這個正則式的字串,都substitute 改成 空白
# <  > ,  + 至少出現1次的 
# [^>] 排除 >
def rm_tags(text):
    re_tag = re.compile(r'<[^>]+>')
    return re_tag.sub('', text)


# In[19]:


#以下是練習
re.escape('1234@gmail.com')


# In[9]:


#import os
def read_files(filetype):
    path = "data/aclImdb/"
    file_list=[]

    positive_path=path + filetype+"/pos/"
    for f in os.listdir(positive_path):
        file_list+=[positive_path+f]
    
    negative_path=path + filetype+"/neg/"
    for f in os.listdir(negative_path):
        file_list+=[negative_path+f]
        
    print('read',filetype, 'files: records=',len(file_list))
       
    all_labels = ([1] * 12500 + [0] * 12500)  #因為原來就知道 正評 負評各50%
    
    all_texts  = []
    for fi in file_list:
        with open(fi,encoding='utf8') as file_input:
            all_texts += [rm_tags(" ".join(file_input.readlines()))]
            
    all_texts_without_re  = []
    for fi in file_list:
        with open(fi,encoding='utf8') as file_input:
            all_texts_without_re  += [ file_input.readlines() ]
            
    return all_labels,all_texts, all_texts_without_re 


# In[14]:


#丟一個參數出去給 function read_files, 收三個回來 所以左邊3個參數接收
y_train,train_text, train_text_wo_re =read_files("train")
print(type(y_train), type(train_text), type(train_text_wo_re))


# In[34]:


y_test,test_text, text_text_wo_re = read_files("test")


# In[11]:


#查看某一篇正面評價的影評, 有 re 過 跟沒有re 差別在哪裡
# 有 [ ]  <br /> <br />


# In[24]:


train_text_wo_re[5]


# In[25]:


train_text[5]


# In[13]:


y_train[0]


# In[14]:


#查看負面評價的影評


# In[27]:


train_text[12501]


# In[26]:


y_train[12501]


# # 先讀取所有文章建立字典，限制字典的數量為nb_words=2000

# In[29]:


#from keras.preprocessing.text import Tokenizer
#遍讀25,000篇評論後, 萃取出重要的, 建立 token字典, 假設2000字就好了
token = Tokenizer(num_words=2000) # Tokenizer屬性
token.fit_on_texts(train_text)   #fit_on_texts 讀取多少文章


# In[30]:


print(token.document_count)


# In[31]:


print(token.word_index) #.word_index 專門來查看字典的


# # 將每一篇文章的文字轉換一連串的數字
# #只有在字典中的文字會轉換為數字
# #使用 token 將影評文字轉換為數字list texts_to_sequence

# In[23]:


print(train_text[0])


# In[35]:


# 注意:這裡的method  是 texts_to_sequences , 不是 pad_sequences,
x_train_seq = token.texts_to_sequences(train_text)
x_test_seq  = token.texts_to_sequences(test_text)


# In[36]:


#print(type(x_train_seq)) #list

print(len(x_train_seq))

'''
for x in x_train_seq:
    print(len(x), '\t', x, '\n') #每一個token翻成sequences的數字個數不一 ~~~~
'''
 


# In[27]:


print(x_train_seq[0])
print()
print(len(x_train_seq[0]))


# # 給予參數,採用pad_sequences, 強迫讓轉換後的每個token 數字個數相同

# In[ ]:


# 注意:這裡的method 是 pad_sequences, 不是 texts_to_sequences


# In[25]:


#文章內的文字，轉換為數字後，每一篇的文章地所產生的數字長度都不同，因為要進行類神經網路的訓練，所以每一篇文章所產生的數字長度必須相同
#以下列程式碼為例 maxlen=100，所以每一篇文章轉換為數字 都為100


# In[42]:


x_train = sequence.pad_sequences(x_train_seq, maxlen=100)
x_test  = sequence.pad_sequences(x_test_seq,  maxlen=100)


# In[27]:


#截長補短: 如果文章轉成數字大於所設定的100, pad_sequences處理後，會 truncate前面的數字 ( 106裡面的前6個)


# In[28]:


print('before pad_sequences length=',len(x_train_seq[0]))
print(x_train_seq[0])


# In[29]:


print('after pad_sequences length=',len(x_train[0]))
print(x_train[0])


# In[30]:


#截長補短: 如果文章轉成數字不足100, 經過 pad_sequences處理後，前面幾個會加上0


# In[40]:


print('before pad_sequences length=',len(x_train_seq[6]))
print(x_train_seq[6])


# In[43]:


print('after pad_sequences length=',len(x_train[6]))
print(x_train[6])


# # 資料預處理 
# # 其實有25,000字,但只丟2000下去訓練 這樣比較快

# In[40]:


token = Tokenizer(num_words=2000)
token.fit_on_texts(train_text)


# In[41]:


x_train_seq = token.texts_to_sequences(train_text)
x_test_seq  = token.texts_to_sequences(test_text)


# In[42]:


x_train = sequence.pad_sequences(x_train_seq, maxlen=100)
x_test  = sequence.pad_sequences(x_test_seq,  maxlen=100)


# In[ ]:




