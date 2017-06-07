import re
from nltk.stem import SnowballStemmer
# from string import punctuation
import numpy as np
import pandas as pd
import os
import math
import datetime
import random

###################### 文本标点符号以及词型处理 ######################
def textHandle(text, stem_words=1):

    ############## 去除标点符号 ##############
    ###### 方法 1 ######
    # 不使用这种方式去除标点符号，因为会导致使用标点符号分割的词连在一起
    # 例如： springstone.com --> springstonecom
    # text = ''.join([c for c in text if c not in punctuation])
    ###### 方法 2 ######
    # 使用这种方式去除标点符号以及非英文词汇
    text = re.sub(r"[^A-Za-z0-9]", " ", text)

    ############## 特殊词以及缩写处理 ##############
    text = re.sub(r"what's", " what is ", text)
    text = re.sub(r"What's", " what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", " cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"I'm", " I am ", text)
    text = re.sub(r" m ", " am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\0k ", "0000 ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e-mail", " email ", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r" quikly ", " quickly ", text)
    text = re.sub(r" usa ", " America ", text)
    text = re.sub(r" USA ", " America ", text)
    text = re.sub(r" u s ", " America ", text)
    text = re.sub(r" uk ", " England ", text)
    text = re.sub(r" UK ", " England ", text)
    text = re.sub(r" imrovement ", " improvement ", text)
    text = re.sub(r" intially ", " initially ", text)
    text = re.sub(r" dms ", " direct messages ", text)  
    text = re.sub(r" demonitization ", " demonetization ", text) 
    text = re.sub(r" actived ", " active ", text)
    text = re.sub(r" kms ", " kilometers ", text)
    text = re.sub(r" KMs ", " kilometers ", text)
    text = re.sub(r" cs ", " computer science ", text) 
    text = re.sub(r" upvotes ", " up votes ", text)
    text = re.sub(r"\0rs ", " rs ", text) 
    text = re.sub(r" calender ", " calendar ", text)
    text = re.sub(r" ios ", " operating system ", text)
    text = re.sub(r"programing", " programming ", text)
    text = re.sub(r"bestfriend", " best friend ", text)
    text = re.sub(r"III", "3", text)
    text = re.sub(r"the US ", "America", text)
    text = re.sub(r" J K ", " JK ", text)

    ############## 全部变为小写 ##############
    text = text.lower()
    
    ############## 恢复词干 ##############
    if stem_words:
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)

    return(text)


###################### 获取总词数 ######################
def getNum(data):
    num = 0
    for i in data.index:
        q1 = str(data.get_value(i, 'question1'))
        q2 = str(data.get_value(i, 'question2'))
        q1 = q1.split()
        q2 = q2.split()
        num += len(q1)
        num += len(q2)
    print('Num is : %d' % (num))
    return num

###################### 计算词频以及词所占比率 ######################
def countWord(data, num):
    dic = {}
    count = 0
    for i in data.index:
        count += 1
        q1 = str(data.get_value(i, 'question1'))
        q2 = str(data.get_value(i, 'question2'))
        q1 = q1.split()
        q2 = q2.split()
        q = q1 + q2
        for j in q:
            if j not in dic.keys():
                dic[j] = 1
            else:
                dic[j] += 1
        if count % 10000 == 0:
            print('We have count the frequency of ' + str(count // 10000) + 'w rows of data')
    dic_f = {}
    for i in dic.keys():
        dic_f[i] = dic[i]/num

    return dic, dic_f


###################### subsampling_stop ######################
def subsampling_stop(data, dic):
    dic_p = {}

    stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
                'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
                'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
                'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
                'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
                'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
                'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
                'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
                'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here',
                'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
                'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
                'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now']

    for i in stopwords:
        try:
            dic_p[i] = 1 - math.sqrt(1e-5/dic[i])
        except:
            continue
    count = 0
    for i in data.index:
        count += 1
        #print(data[i])
        q1 = str(data.get_value(i, 'question1')).split()
        q2 = str(data.get_value(i, 'question2')).split()
        for j in dic_p.keys():
            if j in q1:
                if random.random() <= dic_p[j]:
                    q1.remove(j)
            if j in q2:
                if random.random() <= dic_p[j]:
                    q2.remove(j)
        data.loc[i, 'question1'] = ' '.join(q1)
        data.loc[i, 'question2'] = ' '.join(q2)
        if count % 100 == 0:
            print(str(count // 100) + 'h rows of data have been dealed')
    return data

###################### subsampling_all ######################
def subsampling_all(data, dic):
    dic_p = {}
    for i in dic.keys():
        dic_p[i] = 1 - math.sqrt(1e-5/dic[i])
    count = 0
    for i in data.index:
        count += 1
        q1 = str(data.get_value(i, 'question1')).split()
        q2 = str(data.get_value(i, 'question2')).split()
        for j in dic_p.keys():
            if j in q1:
                if random.random() <= dic_p[j]:
                    q1.remove(j)
            if j in q2:
                if random.random() <= dic_p[j]:
                    q2.remove(j)
        data.loc[i, 'question1'] = ' '.join(q1)
        data.loc[i, 'question2'] = ' '.join(q2)
        if count % 100 == 0:
            print(str(count // 100) + 'h rows of data have been dealed')
    return data

###################### 将词转变为迭代形式 ######################
class dataIter(object):
    """数据向量化"""
    def __init__(self, data):
        super(dataIter, self).__init__()
        self.data = data

    def __iter__(self):
        count = 0
        for i in self.data.index:
            count += 1
            if count % 1000 == 0:
                print(str(count // 1000) + 'k rows of data have been word2phrase')
            for j in [1,2]:
                q = str(self.data.get_value(i, 'question' + str(j)))
                yield str(q).split()
        

###################### word2parse对数据进行修改 ######################
def word2phrase_deal(data, bigram, trigram):
    count = 0
    for i in data.index:
        count += 1
        data.set_value(i, 'question1' ,' '.join(trigram[bigram[str(data.get_value(i, 'question1')).split()]]))
        data.set_value(i, 'question2', ' '.join(trigram[bigram[str(data.get_value(i, 'question2')).split()]]))
        if count % 1000 == 0:
            print(str(count // 1000) + 'k rows of data have been word2phrase dealed')
    return data
