import warnings
warnings.filterwarnings("ignore")

import gensim
import pandas as pd
import os
import re

# 使用word2vec对文本进行向量化并将结果保存在文件中
class allWord2Vec():
    def __init__(self, type, shape=200, window=5, min_count=0):
        self.type = type
        self.shape = shape
        self.window = window
        self.min_count = min_count

        # 将输入文本转化成可迭代对象
        class sentences_generator():
            def __init__(self, type):
                self.type = type

            def __iter__(self):
                data = pd.read_csv('./data/merged_data.csv')

                if self.type == 'no':
                    count = 0
                    count_s = 0
                    for i in data.index:
                        for j in [1, 2]:
                            count += 1
                            q = str(data.get_value(i, 'question' + str(j))).split()
                            if count % 2000 == 0:
                                count_s += 1
                                print('第' + str(count_s) + '千个数据已经生成向量no完毕')
                            yield q
                elif self.type == 'all':
                    count = 0
                    count_s = 0
                    for i in data.index:
                        for j in [1, 2]:
                            count += 1
                            q = str(data.get_value(i, 'question' + str(j) + '_all')).split()
                            if count % 2000 == 0:
                                count_s += 1
                                print('第' + str(count_s) + '千个数据已经生成向量all完毕')
                            yield q
                elif self.type == 'stop':
                    count = 0
                    count_s = 0
                    for i in data.index:
                        for j in [1, 2]:
                            count += 1
                            q = str(data.get_value(i, 'question' + str(j) + '_stop')).split()
                            if count % 2000 == 0:
                                count_s += 1
                                print('第' + str(count_s) + '千个数据已经生成向量stop完毕')
                            yield q

        if os.path.exists('./wordvect/' + str(self.shape) + '_size_train_' + self.type):
            model = gensim.models.Word2Vec.load('./wordvect/' + str(self.shape) +'_size_train_' + self.type)
            self.model = model
            print('向量模型已经存在')
        else:
            print('向量模型不存在，即将生成')
            sentences = sentences_generator(self.type)    
            model = gensim.models.Word2Vec(sentences, size=self.shape, workers=8, window = self.window, min_count = self.min_count)
            model.wv.save_word2vec_format('./wordvect/model_' + str(self.shape) +'_size_train_' + self.type)
            model.delete_temporary_training_data()
            model.save('./wordvect/' + str(self.shape) +'_size_train_' + self.type)
            self.model = model

    def getModel(self):
        return self.model