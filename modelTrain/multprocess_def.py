# 数据处理以及多进程

import pandas as pd
import datetime
import multiprocessing
from scipy.spatial import distance
from scipy.stats import spearmanr, kendalltau


from allword2vec import allWord2Vec
import numpy as np
import math

def initData(shape=200, window=5, min_count=0):
    modelvec = allWord2Vec('no', shape, window, min_count).getModel()
    print('向量模型no生成成功')
    modelvec_all = allWord2Vec('all', shape, window, min_count).getModel()
    print('向量模型all生成成功')
    modelvec_stop = allWord2Vec('stop', shape, window, min_count).getModel()
    print('向量模型stop生成成功')
    return [modelvec, modelvec_all, modelvec_stop]

def valueCal(q, index_list, data, modelvec, modelvec_all, modelvec_stop, length, shape):
    while 1:
        for i in index_list:
            list_con = []
            q1 = str(data.get_value(i, 'question1')).split()
            q2 = str(data.get_value(i, 'question2')).split()
             
            q1 = getVect(q1, modelvec, length, shape)
            q2 = getVect(q2, modelvec, length, shape)

            q1_all = str(data.get_value(i, 'question1_all')).split()
            q2_all = str(data.get_value(i, 'question2_all')).split()
            q1_stop = str(data.get_value(i, 'question1_stop')).split()
            q2_stop = str(data.get_value(i, 'question2_stop')).split()

            q1_all = getVect(q1_all, modelvec_all, length, shape)
            q2_all = getVect(q2_all, modelvec_all, length, shape)
            q1_stop = getVect(q1_stop, modelvec_stop, length, shape)
            q2_stop = getVect(q2_stop, modelvec_stop, length, shape)


            list_con.append(q1)
            list_con.append(q2)

            list_con.append(q1_all)
            list_con.append(q2_all)
            list_con.append(q1_stop)
            list_con.append(q2_stop)



            list_con.append(float(data.get_value(i, 'f1')))
            list_con.append(float(data.get_value(i, 'f2')))
            list_con.append(float(data.get_value(i, 'f3')))
            list_con.append(float(data.get_value(i, 'f4_q1')))
            list_con.append(float(data.get_value(i, 'f4_q2')))
            list_con.append(float(data.get_value(i, 'f4_rate')))
            list_con.append(float(data.get_value(i, 'cos_dis_add')))
            list_con.append(float(data.get_value(i, 'cos_dis_map')))
            list_con.append(float(data.get_value(i, 'euc_dis')))
            list_con.append(float(data.get_value(i, 'q_jac_map_dis')))
            list_con.append(float(data.get_value(i, 'q_jac_add_dis')))
            list_con.append(float(data.get_value(i, 'q_mhd_map_dis')))
            list_con.append(float(data.get_value(i, 'q_pearson_map')))
            list_con.append(float(data.get_value(i, 'q_pearson_add')))
            list_con.append(float(data.get_value(i, 'q_spearmanr_map_t')))
            list_con.append(float(data.get_value(i, 'q_spearmanr_add_t')))
            list_con.append(float(data.get_value(i, 'q_spearmanr_map_p')))
            list_con.append(float(data.get_value(i, 'q_spearmanr_add_p')))
            list_con.append(float(data.get_value(i, 'q_kendalltau_map_p')))
            list_con.append(float(data.get_value(i, 'q_kendalltau_add_p')))
            list_con.append(float(data.get_value(i, 'q_kendalltau_map_t')))
            list_con.append(float(data.get_value(i, 'q_kendalltau_add_t')))

            # all and stop

            list_con.append(float(data.get_value(i, 'f1_all')))
            list_con.append(float(data.get_value(i, 'f2_all')))
            list_con.append(float(data.get_value(i, 'f3_all')))
            list_con.append(float(data.get_value(i, 'f4_q1_all')))
            list_con.append(float(data.get_value(i, 'f4_q2_all')))
            list_con.append(float(data.get_value(i, 'f4_rate_all')))
            list_con.append(float(data.get_value(i, 'f1_stop')))
            list_con.append(float(data.get_value(i, 'f2_stop')))
            list_con.append(float(data.get_value(i, 'f3_stop')))
            list_con.append(float(data.get_value(i, 'f4_q1_stop')))
            list_con.append(float(data.get_value(i, 'f4_q2_stop')))
            list_con.append(float(data.get_value(i, 'f4_rate_stop')))

            list_con.append(float(data.get_value(i, 'cos_dis_all_add')))
            list_con.append(float(data.get_value(i, 'cos_dis_all_map')))
            list_con.append(float(data.get_value(i, 'cos_dis_stop_add')))
            list_con.append(float(data.get_value(i, 'cos_dis_stop_map')))

            list_con.append(float(data.get_value(i, 'euc_dis_all')))
            list_con.append(float(data.get_value(i, 'euc_dis_stop')))

            list_con.append(float(data.get_value(i, 'q_all_jac_map_dis')))
            list_con.append(float(data.get_value(i, 'q_all_jac_add_dis')))
            list_con.append(float(data.get_value(i, 'q_stop_jac_map_dis')))
            list_con.append(float(data.get_value(i, 'q_stop_jac_add_dis')))


            list_con.append(float(data.get_value(i, 'q_all_mhd_map_dis')))
            list_con.append(float(data.get_value(i, 'q_stop_mhd_map_dis')))


            list_con.append(float(data.get_value(i, 'q_all_pearson_map')))
            list_con.append(float(data.get_value(i, 'q_all_pearson_add')))
            list_con.append(float(data.get_value(i, 'q_stop_pearson_map')))
            list_con.append(float(data.get_value(i, 'q_stop_pearson_add')))


            list_con.append(float(data.get_value(i, 'q_all_spearmanr_map_t')))
            list_con.append(float(data.get_value(i, 'q_all_spearmanr_add_t')))
            list_con.append(float(data.get_value(i, 'q_stop_spearmanr_map_t')))
            list_con.append(float(data.get_value(i, 'q_stop_spearmanr_add_t')))    


            list_con.append(float(data.get_value(i, 'q_all_spearmanr_map_p')))
            list_con.append(float(data.get_value(i, 'q_all_spearmanr_add_p')))
            list_con.append(float(data.get_value(i, 'q_stop_spearmanr_map_p')))
            list_con.append(float(data.get_value(i, 'q_stop_spearmanr_add_p')))


            list_con.append(float(data.get_value(i, 'q_all_kendalltau_map_p')))
            list_con.append(float(data.get_value(i, 'q_all_kendalltau_add_p')))
            list_con.append(float(data.get_value(i, 'q_stop_kendalltau_map_p')))
            list_con.append(float(data.get_value(i, 'q_stop_kendalltau_add_p')))

            
            list_con.append(float(data.get_value(i, 'q_all_kendalltau_map_t')))
            list_con.append(float(data.get_value(i, 'q_all_kendalltau_add_t')))
            list_con.append(float(data.get_value(i, 'q_stop_kendalltau_map_t')))
            list_con.append(float(data.get_value(i, 'q_stop_kendalltau_add_t')))

            list_con.append(data.get_value(i, 'is_duplicate'))
            while 1:
                try:
                    q.put(list_con)
                    break
                except:
                    continue

def getVect(q, model, length, shape):
    for j in range(len(q)):
        try:
            q[j] = model.wv[q[j]]
        except:
            q[j] = np.zeros(shape=(shape))
    if len(q) > length:
        q = q[:length]
    elif len(q) < length:
        for i in range(length - len(q)):
            q.append(np.zeros(shape=(shape)))
    return q

def dataSamp(frac=0.3):
    data = pd.read_csv('./data/merged_data_withfeature.csv', 
                        names=['id', 'qid1', 'qid2', 'question1_all', 
                                'question2_all', 'question1_stop', 
                                'question2_stop', 'question1', 
                                'question2', 'is_duplicate', 
                                'f1', 'f2', 'f3', 'f4_q1', 
                                'f4_q2', 'f4_rate', 'cos_dis_add', 
                                'cos_dis_map', 'euc_dis', 'q_jac_map_dis', 
                                'q_jac_add_dis', 'q_mhd_map_dis', 'q_pearson_map', 
                                'q_pearson_add', 'q_spearmanr_map_t', 'q_spearmanr_add_t', 
                                'q_spearmanr_map_p', 'q_spearmanr_add_p', 'q_kendalltau_map_p', 
                                'q_kendalltau_add_p', 'q_kendalltau_map_t', 'q_kendalltau_add_t', 
                                'f1_all', 'f2_all', 'f3_all', 'f4_q1_all', 'f4_q2_all', 
                                'f4_rate_all', 'f1_stop', 'f2_stop', 'f3_stop', 
                                'f4_q1_stop', 'f4_q2_stop', 'f4_rate_stop', 'cos_dis_all_add', 
                                'cos_dis_all_map', 'cos_dis_stop_add', 'cos_dis_stop_map', 
                                'euc_dis_all', 'euc_dis_stop', 'q_all_jac_map_dis', 
                                'q_all_jac_add_dis', 'q_stop_jac_map_dis', 'q_stop_jac_add_dis', 
                                'q_all_mhd_map_dis', 'q_stop_mhd_map_dis', 'q_all_pearson_map', 
                                'q_all_pearson_add', 'q_stop_pearson_map', 'q_stop_pearson_add', 
                                'q_all_spearmanr_map_t', 'q_all_spearmanr_add_t', 
                                'q_stop_spearmanr_map_t', 'q_stop_spearmanr_add_t', 
                                'q_all_spearmanr_map_p', 'q_all_spearmanr_add_p', 
                                'q_stop_spearmanr_map_p', 'q_stop_spearmanr_add_p', 
                                'q_all_kendalltau_map_p', 'q_all_kendalltau_add_p', 
                                'q_stop_kendalltau_map_p', 'q_stop_kendalltau_add_p', 
                                'q_all_kendalltau_map_t', 'q_all_kendalltau_add_t', 
                                'q_stop_kendalltau_map_t', 'q_stop_kendalltau_add_t'])

    test1 = data[data.is_duplicate==0].sample(frac=frac)
    test2 = data[data.is_duplicate==1].sample(frac=frac)

    train = data.drop(test1.index).drop(test2.index)
    test = pd.concat([test1, test2],ignore_index=True)

    test.to_csv('./data/test_merged_' + str(int(frac*10)) + '.csv', index = 0)
    train.to_csv('./data/train_merged_' + str(int(10 - frac*10)) + '.csv', index = 0)
    print('测试集训练集抽取完毕')

def justDoitData(type, frac, patch_size=32, length=36, shape=200, window=5, min_count=0, worker=8):
    # while 1:
        
    multiprocessing.freeze_support()
    pool = multiprocessing.Pool()
    if type == 'train':
        # data = pd.read_csv('./data/train_merged_' + str(int(10 - frac*10)) + '.csv')
        data = pd.read_csv('./data/merged_data_withfeature.csv', 
                            names=['id', 'qid1', 'qid2', 'question1_all', 
                                'question2_all', 'question1_stop', 
                                'question2_stop', 'question1', 
                                'question2', 'is_duplicate', 
                                'f1', 'f2', 'f3', 'f4_q1', 
                                'f4_q2', 'f4_rate', 'cos_dis_add', 
                                'cos_dis_map', 'euc_dis', 'q_jac_map_dis', 
                                'q_jac_add_dis', 'q_mhd_map_dis', 'q_pearson_map', 
                                'q_pearson_add', 'q_spearmanr_map_t', 'q_spearmanr_add_t', 
                                'q_spearmanr_map_p', 'q_spearmanr_add_p', 'q_kendalltau_map_p', 
                                'q_kendalltau_add_p', 'q_kendalltau_map_t', 'q_kendalltau_add_t', 
                                'f1_all', 'f2_all', 'f3_all', 'f4_q1_all', 'f4_q2_all', 
                                'f4_rate_all', 'f1_stop', 'f2_stop', 'f3_stop', 
                                'f4_q1_stop', 'f4_q2_stop', 'f4_rate_stop', 'cos_dis_all_add', 
                                'cos_dis_all_map', 'cos_dis_stop_add', 'cos_dis_stop_map', 
                                'euc_dis_all', 'euc_dis_stop', 'q_all_jac_map_dis', 
                                'q_all_jac_add_dis', 'q_stop_jac_map_dis', 'q_stop_jac_add_dis', 
                                'q_all_mhd_map_dis', 'q_stop_mhd_map_dis', 'q_all_pearson_map', 
                                'q_all_pearson_add', 'q_stop_pearson_map', 'q_stop_pearson_add', 
                                'q_all_spearmanr_map_t', 'q_all_spearmanr_add_t', 
                                'q_stop_spearmanr_map_t', 'q_stop_spearmanr_add_t', 
                                'q_all_spearmanr_map_p', 'q_all_spearmanr_add_p', 
                                'q_stop_spearmanr_map_p', 'q_stop_spearmanr_add_p', 
                                'q_all_kendalltau_map_p', 'q_all_kendalltau_add_p', 
                                'q_stop_kendalltau_map_p', 'q_stop_kendalltau_add_p', 
                                'q_all_kendalltau_map_t', 'q_all_kendalltau_add_t', 
                                'q_stop_kendalltau_map_t', 'q_stop_kendalltau_add_t'])
    else:
        data = pd.read_csv('./data/test_merged_' + str(int(frac*10)) + '.csv')
    model = initData(shape, window, min_count)
    index_list = []
    manager = multiprocessing.Manager()
    q = manager.Queue(patch_size*100)
    count = 0

    alldata = data

    index_list.append(data.sample(frac=(1/worker)).index)
    for i in range(worker-1):
        data = data.drop(index_list[-1])
        index_list.append(data.sample(frac=(1/(worker-1-i))).index)

    for i in range(worker):
        print('添加第'+str(i)+'个进程 : ' + str(datetime.datetime.now()))
        pool.apply_async(valueCal, args=(q, index_list[i], alldata, model[0], model[1], model[2], length, shape))
        print('第'+str(i)+'个进程添加完毕 : ' + str(datetime.datetime.now()))
    pool.close()
    print('进程池关闭成功 : ' + str(datetime.datetime.now()))
    get_list = []
    suml = 0
    cd = 0
    count = 0

    # for ide in alldata.index:
    while 1:
        count += 1
        if count%1000 == 0:
            print('\n   ' + str(count//1000) + 'k 行数据已经被处理, 正样本比例 ：' + str(suml / cd))
        # if count > len(alldata):
        #     break

        get_list.append(q.get(True))
        if len(get_list) == patch_size:
            temp_list = get_list
            get_list = []
            suml += sum([i[-1] for i in temp_list])
            cd += patch_size

            # print('\n\n' + str(cd) + '\n')

            yield ({ 'q_vec_1_input': np.asarray([i[0] for i in temp_list]), 'q_vec_2_input': np.asarray([i[1] for i in temp_list]),
                    'q_vec_1_all_input': np.asarray([i[2] for i in temp_list]), 'q_vec_2_all_input': np.asarray([i[3] for i in temp_list]),
                    'q_vec_1_stop_input': np.asarray([i[4] for i in temp_list]), 'q_vec_2_stop_input': np.asarray([i[5] for i in temp_list]),
                    'input_feature': np.asarray([i[6:-1] for i in temp_list])}, {'output_dense': np.asarray([i[-1] for i in temp_list])})




def testvalueCal(q, data, modelvec, modelvec_all, modelvec_stop, length, shape):
    for i in data.index:
        list_con = []
        q1 = str(data.get_value(i, 'question1')).split()
        q2 = str(data.get_value(i, 'question2')).split()
         
        q1 = getVect(q1, modelvec, length, shape)
        q2 = getVect(q2, modelvec, length, shape)

        q1_all = str(data.get_value(i, 'question1_all')).split()
        q2_all = str(data.get_value(i, 'question2_all')).split()
        q1_stop = str(data.get_value(i, 'question1_stop')).split()
        q2_stop = str(data.get_value(i, 'question2_stop')).split()

        q1_all = getVect(q1_all, modelvec_all, length, shape)
        q2_all = getVect(q2_all, modelvec_all, length, shape)
        q1_stop = getVect(q1_stop, modelvec_stop, length, shape)
        q2_stop = getVect(q2_stop, modelvec_stop, length, shape)


        list_con.append(q1)
        list_con.append(q2)

        list_con.append(q1_all)
        list_con.append(q2_all)
        list_con.append(q1_stop)
        list_con.append(q2_stop)



        list_con.append(float(data.get_value(i, 'f1')))
        list_con.append(float(data.get_value(i, 'f2')))
        list_con.append(float(data.get_value(i, 'f3')))
        list_con.append(float(data.get_value(i, 'f4_q1')))
        list_con.append(float(data.get_value(i, 'f4_q2')))
        list_con.append(float(data.get_value(i, 'f4_rate')))
        list_con.append(float(data.get_value(i, 'cos_dis_add')))
        list_con.append(float(data.get_value(i, 'cos_dis_map')))
        list_con.append(float(data.get_value(i, 'euc_dis')))
        list_con.append(float(data.get_value(i, 'q_jac_map_dis')))
        list_con.append(float(data.get_value(i, 'q_jac_add_dis')))
        list_con.append(float(data.get_value(i, 'q_mhd_map_dis')))
        list_con.append(float(data.get_value(i, 'q_pearson_map')))
        list_con.append(float(data.get_value(i, 'q_pearson_add')))
        list_con.append(float(data.get_value(i, 'q_spearmanr_map_t')))
        list_con.append(float(data.get_value(i, 'q_spearmanr_add_t')))
        list_con.append(float(data.get_value(i, 'q_spearmanr_map_p')))
        list_con.append(float(data.get_value(i, 'q_spearmanr_add_p')))
        list_con.append(float(data.get_value(i, 'q_kendalltau_map_p')))
        list_con.append(float(data.get_value(i, 'q_kendalltau_add_p')))
        list_con.append(float(data.get_value(i, 'q_kendalltau_map_t')))
        list_con.append(float(data.get_value(i, 'q_kendalltau_add_t')))

        # all and stop

        list_con.append(float(data.get_value(i, 'f1_all')))
        list_con.append(float(data.get_value(i, 'f2_all')))
        list_con.append(float(data.get_value(i, 'f3_all')))
        list_con.append(float(data.get_value(i, 'f4_q1_all')))
        list_con.append(float(data.get_value(i, 'f4_q2_all')))
        list_con.append(float(data.get_value(i, 'f4_rate_all')))
        list_con.append(float(data.get_value(i, 'f1_stop')))
        list_con.append(float(data.get_value(i, 'f2_stop')))
        list_con.append(float(data.get_value(i, 'f3_stop')))
        list_con.append(float(data.get_value(i, 'f4_q1_stop')))
        list_con.append(float(data.get_value(i, 'f4_q2_stop')))
        list_con.append(float(data.get_value(i, 'f4_rate_stop')))

        list_con.append(float(data.get_value(i, 'cos_dis_all_add')))
        list_con.append(float(data.get_value(i, 'cos_dis_all_map')))
        list_con.append(float(data.get_value(i, 'cos_dis_stop_add')))
        list_con.append(float(data.get_value(i, 'cos_dis_stop_map')))

        list_con.append(float(data.get_value(i, 'euc_dis_all')))
        list_con.append(float(data.get_value(i, 'euc_dis_stop')))

        list_con.append(float(data.get_value(i, 'q_all_jac_map_dis')))
        list_con.append(float(data.get_value(i, 'q_all_jac_add_dis')))
        list_con.append(float(data.get_value(i, 'q_stop_jac_map_dis')))
        list_con.append(float(data.get_value(i, 'q_stop_jac_add_dis')))


        list_con.append(float(data.get_value(i, 'q_all_mhd_map_dis')))
        list_con.append(float(data.get_value(i, 'q_stop_mhd_map_dis')))


        list_con.append(float(data.get_value(i, 'q_all_pearson_map')))
        list_con.append(float(data.get_value(i, 'q_all_pearson_add')))
        list_con.append(float(data.get_value(i, 'q_stop_pearson_map')))
        list_con.append(float(data.get_value(i, 'q_stop_pearson_add')))


        list_con.append(float(data.get_value(i, 'q_all_spearmanr_map_t')))
        list_con.append(float(data.get_value(i, 'q_all_spearmanr_add_t')))
        list_con.append(float(data.get_value(i, 'q_stop_spearmanr_map_t')))
        list_con.append(float(data.get_value(i, 'q_stop_spearmanr_add_t')))    


        list_con.append(float(data.get_value(i, 'q_all_spearmanr_map_p')))
        list_con.append(float(data.get_value(i, 'q_all_spearmanr_add_p')))
        list_con.append(float(data.get_value(i, 'q_stop_spearmanr_map_p')))
        list_con.append(float(data.get_value(i, 'q_stop_spearmanr_add_p')))


        list_con.append(float(data.get_value(i, 'q_all_kendalltau_map_p')))
        list_con.append(float(data.get_value(i, 'q_all_kendalltau_add_p')))
        list_con.append(float(data.get_value(i, 'q_stop_kendalltau_map_p')))
        list_con.append(float(data.get_value(i, 'q_stop_kendalltau_add_p')))

        
        list_con.append(float(data.get_value(i, 'q_all_kendalltau_map_t')))
        list_con.append(float(data.get_value(i, 'q_all_kendalltau_add_t')))
        list_con.append(float(data.get_value(i, 'q_stop_kendalltau_map_t')))
        list_con.append(float(data.get_value(i, 'q_stop_kendalltau_add_t')))

        list_con.append(float(data.get_value(i, 'test_id')))


        q.put(list_con)

def createlog(trainedepochs, epochs, input_shape, input_length, patch_size, frac, train_batch, test_batch, lstm_num, trainnote):
    return '         parameter       value\n    1    trainedepochs   '+str(trainedepochs)+'\n    2    epochs          '+str(epochs)+'\n    3    input_length    '+str(input_length)+'\n    4    input_shape     '+str(input_shape)+'\n    5    batch_size      '+str(patch_size)+'\n    6    frac            '+str(frac)+'\n    7    train_batch     '+str(train_batch)+'\n    8    test_batch      '+str(test_batch)+'\n    9    lstm_num        '+str(lstm_num)+'\n    10   note            '+str(trainnote)+'\n'

# if __name__ == '__main__':
#     for i in justDoitData('train', 0.3, 128, 36, 50, worker=8):
#         print(i[1])
