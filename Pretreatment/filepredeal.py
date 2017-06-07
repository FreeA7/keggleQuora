import pandas as pd
import numpy as np
import re
import json
import filepredeal_def
from gensim.models import phrases

input_shape = 20
input_length = 36
frac = 0.3

d = 1

###################### 数据 ######################
# 将乱序后的数据分成8份
# data = pd.read_csv('./data/train.csv')
# for i in range(8):
# 	data_temp = data.sample(frac=(1/(8-i)), replace = False)
# 	data = data.drop(data_temp.index)
# 	data_temp.to_csv('./data/temp/data_train_be_pre_' + str(i+1) + '.csv', encoding = 'utf-8', index = 0)

###################### 文本的规整，去缩写，恢复词干 ######################
# 手动15个多进程读取8份文件处理后输出到8个csv之中
# print (str(d) + ' read')
# data = pd.read_csv('./data/temp/data_train_be_pre_' + str(d) + '.csv')
# count = 0
# for i in data.index:
#     count += 1
#     try:
#     	q1 = filepredeal_def.textHandle(data.get_value(i, 'question1'))
#     	q2 = filepredeal_def.textHandle(data.get_value(i, 'question2'))
#     except:
#     	continue
#     data.loc[i, 'question1'] = q1
#     data.loc[i, 'question2'] = q2
#     if count % 100 == 0:
#         print(str(count // 100) + 'h rows of data have been dealed')
# print('All data has been dealed')
# data.to_csv('./data/temp/data_train_after_pre_' + str(d) + '.csv', index = 0)


###################### 数据合并 ######################
# 读取处理后的8个csv，并合并为一个csv进行存储使用
# data = []
# for i in range(8):
#     if i == 0:
#         data = pd.read_csv('./data/temp/data_train_after_pre_' + str(i+1) + '.csv')
#         continue
#     data = pd.concat([data, pd.read_csv('./data/temp/data_train_after_pre_' + str(i+1) + '.csv')],ignore_index=True)
# data.to_csv('./data/data_train_after_pre.csv', index = 0)

###################### word2phrase ######################
# data = []
# data = pd.read_csv('./data/data_train_after_pre.csv')
# iter_ob = filepredeal_def.dataIter(data)
# phrases_done = phrases.Phrases(iter_ob)
# print('第一次word2phrase生成成功')
# bigram = phrases.Phraser(phrases_done)
# iter_ob = filepredeal_def.dataIter(data)
# trigram = phrases.Phrases(bigram[iter_ob])
# print('第二次word2phrase生成成功')
# bigram.save('./data/word2phrase_sub_bigram')
# trigram.save('./data/word2phrase_sub_trigram')
# print('word2phrase生成模型保存成功')

###################### 加载 word2phrase all 模型处理数据 ######################
# data = pd.read_csv('./data/data_train_after_pre.csv')
# bigram = phrases.Phraser.load('./data/word2phrase_sub_bigram')
# trigram = phrases.Phrases.load('./data/word2phrase_sub_trigram')
# data = filepredeal_def.word2phrase_deal(data, bigram, trigram)
# data.to_csv('./data/data_train_after_word2praes.csv', index = 0)

###################### 获取词频和总词数并保存为json ######################
# data = pd.read_csv('./data/data_train_after_word2praes.csv')
# num = filepredeal_def.getNum(data)
# dic, dic_f = filepredeal_def.countWord(data, num)
# f = open('./data/dic/num','w')
# f.write(str(num))
# f.close()
# with open('./data/dic/dic.json','w') as outfile:  
#    json.dump(dic,outfile,ensure_ascii=False)  
#    outfile.write('\n')
# with open('./data/dic/dic_f.json','w') as outfile:  
#    json.dump(dic_f,outfile,ensure_ascii=False)  
#    outfile.write('\n')

###################### 将数据分为8份进行处理 ######################
# data = pd.read_csv('./data/data_train_after_word2praes.csv')
# for i in range(8):
#    data_temp = data.sample(frac=(1/(8-i)))
#    index = data_temp.index
#    data = data.drop(index)
#    data_temp.to_csv('./data/temp/data_train_be_sub_' + str(i+1) + '.csv', encoding = 'utf-8', index=0)

###################### 8份数据手动多进程进行停用词处理 ######################
# print(str(d) + ' stop')
# data = pd.read_csv('./data/temp/data_train_be_sub_' + str(d) + '.csv')
# dic = json.load(open('./data/dic/dic.json'))
# dic_f = json.load(open('./data/dic/dic_f.json'))
# f = open('./data/dic/num','r')
# num = int(f.readline())
# f.close()
# data = filepredeal_def.subsampling_stop(data, dic_f)
# data.to_csv('./data/temp/data_train_after_sub_stop_' + str(d) + '.csv', encoding = 'utf-8', index=0)

###################### 8份数据手动多进程进行所有词处理 ######################
# print(str(d) + ' all')
# data = pd.read_csv('./data/temp/data_train_be_sub_' + str(d) + '.csv')
# dic = json.load(open('./data/dic/dic.json'))
# dic_f = json.load(open('./data/dic/dic_f.json'))
# f = open('./data/dic/num','r')
# num = int(f.readline())
# f.close()
# data = filepredeal_def.subsampling_all(data, dic_f)
# data.to_csv('./data/temp/data_train_after_sub_all_' + str(d) + '.csv', encoding = 'utf-8', index=0)

###################### 8份数据合并处理 ######################
# data = []
# for i in range(8):
#     if i == 0:
#         data = pd.read_csv('./data/temp/data_train_after_sub_all_' + str(i+1) + '.csv')
#         continue
#     data = pd.concat([data, pd.read_csv('./data/temp/data_train_after_sub_all_' + str(i+1) + '.csv')],ignore_index=True)
# data.to_csv('./data/data_train_after_sub_all_pre.csv', encoding = 'utf-8', columns=['id', 'qid1', 'qid2', 'question1', 'question2', 'is_duplicate'], index=0)
# data = []
# for i in range(8):
#     if i == 0:
#         data = pd.read_csv('./data/temp/data_train_after_sub_stop_' + str(i+1) + '.csv')
#         continue
#     data = pd.concat([data, pd.read_csv('./data/temp/data_train_after_sub_stop_' + str(i+1) + '.csv')],ignore_index=True)
# data.to_csv('./data/data_train_after_sub_stop_pre.csv', encoding = 'utf-8', columns=['id', 'qid1', 'qid2', 'question1', 'question2', 'is_duplicate'], index=0)

###################### 数据三合一 正负平衡打乱顺序 ######################
# data1 = pd.read_csv('./data/data_train_after_sub_all_pre.csv')
# data2 = pd.read_csv('./data/data_train_after_sub_stop_pre.csv')
# data3 = pd.read_csv('./data/data_train_after_pre.csv')
# data = pd.merge(data1, data2, on=['id', 'qid1', 'qid2', 'is_duplicate'])
# data = pd.merge(data, data3, on=['id', 'qid1', 'qid2', 'is_duplicate'])
# duplicate_percentage = data['is_duplicate'].sum() / len(data)
# print('The number of question is ' + str(len(data)))
# print('The number of duplicated question is ' + str(data['is_duplicate'].sum()))
# print('The number of unduplicated question is ' + str(len(data) - data['is_duplicate'].sum()))
# print('The percentage of duplicated question is ' + str(duplicate_percentage))
# data = data.sample(frac=1, replace=False)
# print('All the observations have been shuffled')
# del data['Unnamed: 0_x']
# del data['Unnamed: 0_y']
# del data['Unnamed: 0']
# data.columns = ['id', 'qid1', 'qid2', 'question1_all', 'question2_all', 'is_duplicate','question1_stop', 'question2_stop', 'question1', 'question2']
# data.to_csv('./data/merged_data.csv', index = 0)
