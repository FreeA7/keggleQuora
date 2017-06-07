import pandas as pd
import numpy as np
import multiprocessing
from scipy.spatial import distance
from scipy.stats import spearmanr, kendalltau
from allword2vec import allWord2Vec
import math



def initData(shape=200, window=5, min_count=0):
    modelvec = allWord2Vec('no', shape, window, min_count).getModel()
    print('向量模型no生成成功')
    modelvec_all = allWord2Vec('all', shape, window, min_count).getModel()
    print('向量模型all生成成功')
    modelvec_stop = allWord2Vec('stop', shape, window, min_count).getModel()
    print('向量模型stop生成成功')
    return [modelvec, modelvec_all, modelvec_stop]


def getfeature(worker, length=36, shape=200, window=5, min_count=0):
    multiprocessing.freeze_support()
    pool = multiprocessing.Pool()
    data = pd.read_csv('./data/merged_data.csv')
    print('数据读取成功')
    temp_index = []
    for i in range(worker):
        temp_index.append(data.sample(frac=(1/(worker-i)), replace = False).index) 
        data = data.drop(temp_index[-1])
    data = pd.read_csv('./data/merged_data.csv')
    model = initData(shape, window, min_count)
    manager = multiprocessing.Manager()
    q = manager.Queue(60)
    for i in range(worker):
        print('开始创建第' + str(i+1) + '个进程')
        pool.apply_async(valueCal, args=(q, temp_index[i], data, model[0], model[1], model[2], length, shape))
        print('创建第' + str(i+1) + '个进程成功')
    pool.close()
    count = 0
    f = open('./data/merged_data_withfeature.csv', 'w')
    while 1:
        count += 1
        if count % 100 == 0:
            print(str(count//100) + 'h 行数据已经被处理')
            f.flush()
        row = q.get(True)
        row = [str(i) for i in row]
        f.write(','.join(row) + '\n')
    f.close()




def valueCal(q, index_list, data, modelvec, modelvec_all, modelvec_stop, length, shape):
    for i in index_list:
        list_con = []
        q1 = str(data.get_value(i, 'question1')).split()
        q2 = str(data.get_value(i, 'question2')).split()
        f1 = getF1_union(q1, q2)
        f2 = getF2_inter(q1, q2)
        f3 = getF3_sum(q1, q2)
        f4_q1 = len(q1)
        f4_q2 = len(q2)
        f4_rate = f4_q1/f4_q2   
        q1 = getVect(q1, modelvec, length, shape)
        q2 = getVect(q2, modelvec, length, shape)
        cos_dis_add = getCosDis_add(q1, q2)
        cos_dis_map = getCosDis_map(q1, q2)
        euc_dis = getEucDis(q1, q2)    


        # all and stop


        q1_all = str(data.get_value(i, 'question1_all')).split()
        q2_all = str(data.get_value(i, 'question2_all')).split()
        q1_stop = str(data.get_value(i, 'question1_stop')).split()
        q2_stop = str(data.get_value(i, 'question2_stop')).split()

        f1_all = getF1_union(q1_all, q2_all)
        f2_all = getF2_inter(q1_all, q2_all)
        f3_all = getF3_sum(q1_all, q2_all)
        f4_q1_all = len(q1_all)
        f4_q2_all = len(q2_all)
        f4_rate_all = f4_q1_all/f4_q2_all            
        f1_stop = getF1_union(q1_stop, q2_stop)
        f2_stop = getF2_inter(q1_stop, q2_stop)
        f3_stop = getF3_sum(q1_stop, q2_stop)
        f4_q1_stop = len(q1_stop)
        f4_q2_stop = len(q2_stop)
        f4_rate_stop = f4_q1_stop/f4_q2_stop            

        q1_all = getVect(q1_all, modelvec_all, length, shape)
        q2_all = getVect(q2_all, modelvec_all, length, shape)
        q1_stop = getVect(q1_stop, modelvec_stop, length, shape)
        q2_stop = getVect(q2_stop, modelvec_stop, length, shape)

        cos_dis_all_add = getCosDis_add(q1_all, q2_all)
        cos_dis_all_map = getCosDis_map(q1_all, q2_all)
        cos_dis_stop_add = getCosDis_add(q1_stop, q2_stop)
        cos_dis_stop_map = getCosDis_map(q1_stop, q2_stop)
        
        euc_dis_all = getEucDis(q1_all, q2_all)
        euc_dis_stop = getEucDis(q1_stop, q2_stop)

        

        q1_map = vecmap(q1)
        q2_map = vecmap(q2)
        q1_add = vecadd(q1)
        q2_add = vecadd(q2)
        q_jac_map_dis = distance.jaccard(q1_map, q2_map)
        q_jac_add_dis = distance.jaccard(q1_add, q2_add)
        q_mhd_map_dis = distance.mahalanobis(q1_map, q2_map, np.linalg.pinv(np.cov(np.vstack((q1_map,q2_map)).T)))
        q_pearson_map = cal_pearson(q1_map, q2_map)
        q_pearson_add = cal_pearson(q1_add, q2_add)
        q_spearmanr_map_t, q_spearmanr_map_p = spearmanr(q1_map, q2_map)
        q_spearmanr_add_t, q_spearmanr_add_p = spearmanr(q1_add, q2_add)
        q_kendalltau_map_t, q_kendalltau_map_p = kendalltau(q1_map, q2_map)
        q_kendalltau_add_t, q_kendalltau_add_p = kendalltau(q1_add, q2_add)



        # all and stop

        q1_all_map = vecmap(q1_all)
        q2_all_map = vecmap(q2_all)
        q1_all_add = vecadd(q1_all)
        q2_all_add = vecadd(q2_all)
        q1_stop_map = vecmap(q1_stop)
        q2_stop_map = vecmap(q2_stop)
        q1_stop_add = vecadd(q1_stop)
        q2_stop_add = vecadd(q2_stop)                        


        q_all_jac_map_dis = distance.jaccard(q1_all_map, q2_all_map)
        q_all_jac_add_dis = distance.jaccard(q1_all_add, q2_all_add)
        q_stop_jac_map_dis = distance.jaccard(q1_stop_map, q2_stop_map)
        q_stop_jac_add_dis = distance.jaccard(q1_stop_add, q2_stop_add)


        q_all_mhd_map_dis = distance.mahalanobis(q1_all_map, q2_all_map, np.linalg.pinv(np.cov(np.vstack((q1_all_map,q2_all_map)).T)))
        q_stop_mhd_map_dis = distance.mahalanobis(q1_stop_map, q2_stop_map, np.linalg.pinv(np.cov(np.vstack((q1_stop_map,q2_stop_map)).T)))


        q_all_pearson_map = cal_pearson(q1_all_map, q2_all_map)
        q_all_pearson_add = cal_pearson(q1_all_add, q2_all_add)
        q_stop_pearson_map = cal_pearson(q1_stop_map, q2_stop_map)
        q_stop_pearson_add = cal_pearson(q1_stop_add, q2_stop_add)


        q_all_spearmanr_map_t, q_all_spearmanr_map_p = spearmanr(q1_all_map, q2_all_map)
        q_all_spearmanr_add_t, q_all_spearmanr_add_p = spearmanr(q1_all_add, q2_all_add)
        q_stop_spearmanr_map_t, q_stop_spearmanr_map_p = spearmanr(q1_stop_map, q2_stop_map)
        q_stop_spearmanr_add_t, q_stop_spearmanr_add_p = spearmanr(q1_stop_add, q2_stop_add)

        
        q_all_kendalltau_map_t, q_all_kendalltau_map_p = kendalltau(q1_all_map, q2_all_map)
        q_all_kendalltau_add_t, q_all_kendalltau_add_p = kendalltau(q1_all_add, q2_all_add)
        q_stop_kendalltau_map_t, q_stop_kendalltau_map_p = kendalltau(q1_stop_map, q2_stop_map)
        q_stop_kendalltau_add_t, q_stop_kendalltau_add_p = kendalltau(q1_stop_add, q2_stop_add)




        # list_con.append(q1)
        # list_con.append(q2)

        # all and stop

        # list_con.append(q1_all)
        # list_con.append(q2_all)
        # list_con.append(q1_stop)
        # list_con.append(q2_stop)

        list_con.append(data.get_value(i, 'id'))
        list_con.append(data.get_value(i, 'qid1'))
        list_con.append(data.get_value(i, 'qid2'))
        list_con.append(data.get_value(i, 'question1_all'))
        list_con.append(data.get_value(i, 'question2_all'))
        list_con.append(data.get_value(i, 'question1_stop'))
        list_con.append(data.get_value(i, 'question2_stop'))
        list_con.append(data.get_value(i, 'question1'))
        list_con.append(data.get_value(i, 'question2'))
        list_con.append(data.get_value(i, 'is_duplicate'))

        list_con.append(f1)
        list_con.append(f2)
        list_con.append(f3)
        list_con.append(f4_q1)
        list_con.append(f4_q2)
        list_con.append(f4_rate)
        list_con.append(cos_dis_add)
        list_con.append(cos_dis_map)
        list_con.append(euc_dis)
        list_con.append(q_jac_map_dis)
        list_con.append(q_jac_add_dis)
        list_con.append(q_mhd_map_dis)
        list_con.append(q_pearson_map)
        list_con.append(q_pearson_add)
        list_con.append(q_spearmanr_map_t)
        list_con.append(q_spearmanr_add_t)
        list_con.append(q_spearmanr_map_p)
        list_con.append(q_spearmanr_add_p)
        list_con.append(q_kendalltau_map_p)
        list_con.append(q_kendalltau_add_p)
        list_con.append(q_kendalltau_map_t)
        list_con.append(q_kendalltau_add_t)

        # all and stop

        list_con.append(f1_all)
        list_con.append(f2_all)
        list_con.append(f3_all)
        list_con.append(f4_q1_all)
        list_con.append(f4_q2_all)
        list_con.append(f4_rate_all)
        list_con.append(f1_stop)
        list_con.append(f2_stop)
        list_con.append(f3_stop)
        list_con.append(f4_q1_stop)
        list_con.append(f4_q2_stop)
        list_con.append(f4_rate_stop)

        list_con.append(cos_dis_all_add)
        list_con.append(cos_dis_all_map)
        list_con.append(cos_dis_stop_add)
        list_con.append(cos_dis_stop_map)

        list_con.append(euc_dis_all)
        list_con.append(euc_dis_stop)

        list_con.append(q_all_jac_map_dis)
        list_con.append(q_all_jac_add_dis)
        list_con.append(q_stop_jac_map_dis)
        list_con.append(q_stop_jac_add_dis)


        list_con.append(q_all_mhd_map_dis)
        list_con.append(q_stop_mhd_map_dis)


        list_con.append(q_all_pearson_map)
        list_con.append(q_all_pearson_add)
        list_con.append(q_stop_pearson_map)
        list_con.append(q_stop_pearson_add)


        list_con.append(q_all_spearmanr_map_t)
        list_con.append(q_all_spearmanr_add_t)
        list_con.append(q_stop_spearmanr_map_t)
        list_con.append(q_stop_spearmanr_add_t)     


        list_con.append(q_all_spearmanr_map_p)
        list_con.append(q_all_spearmanr_add_p)
        list_con.append(q_stop_spearmanr_map_p)
        list_con.append(q_stop_spearmanr_add_p) 


        list_con.append(q_all_kendalltau_map_p)
        list_con.append(q_all_kendalltau_add_p)
        list_con.append(q_stop_kendalltau_map_p)
        list_con.append(q_stop_kendalltau_add_p)

        
        list_con.append(q_all_kendalltau_map_t)
        list_con.append(q_all_kendalltau_add_t)
        list_con.append(q_stop_kendalltau_map_t)
        list_con.append(q_stop_kendalltau_add_t)

        
        while 1:
            try:
                q.put(list_con)
                break
            except:
                continue

def multiply(a,b):
    #a,b两个列表的数据一一对应相乘之后求和
    sum_ab=0.0
    for i in range(len(a)):
        temp=a[i]*b[i]
        sum_ab+=temp
    return sum_ab

def cal_pearson(x,y):
    n=len(x)
    #求x_list、y_list元素之和
    sum_x=sum(x)
    sum_y=sum(y)
    #求x_list、y_list元素乘积之和
    sum_xy=multiply(x,y)
    #求x_list、y_list的平方和
    sum_x2 = sum([pow(i,2) for i in x])
    sum_y2 = sum([pow(j,2) for j in y])
    molecular=sum_xy-(float(sum_x)*float(sum_y)/n)
    #计算Pearson相关系数，molecular为分子，denominator为分母
    denominator=math.sqrt((sum_x2-float(sum_x**2)/n)*(sum_y2-float(sum_y**2)/n))
    return molecular/denominator


def vecmap(q):
    li_long = [i*0.0 for i in q[0]]
    for lo in q:
        count = 0
        for los in zip(li_long, list(lo)):
            li_long[count] = los[0] + los[1]
            count += 1  
    return list(li_long)

def vecadd(q):
    li_long = []
    for lo in q:
        li_long += list(lo)
    return list(li_long)

def getCosDis_add(q1, q2):
    li_long_1 = []
    sum_1 = 0
    for lo in q1:
        li_long_1 += list(lo)
    for lo in li_long_1:
        sum_1 += lo**2
    sum_1 = math.sqrt(sum_1)
    li_long_1 = np.asarray(li_long_1)
    
    li_long_2 = []
    sum_2 = 0
    for lo in q2:
        li_long_2 += list(lo)
    for lo in li_long_2:
        sum_2 += lo**2
    sum_2 = math.sqrt(sum_2)
    li_long_2 = np.asarray(li_long_2)

    return np.dot(li_long_1, li_long_2) / (sum_1*sum_2)

def getCosDis_map(q1, q2):
    li_long_1 = [i*0.0 for i in q1[0]]
    sum_1 = 0    
    for lo in q1:
        count_l = 0
        for los in zip(li_long_1, list(lo)):
            li_long_1[count_l] = los[0] + los[1]
            count_l += 1
    for lo in li_long_1:
        sum_1 += lo**2
    sum_1 = math.sqrt(sum_1)
    li_long_1 = np.asarray(li_long_1)

    li_long_2 = [i*0.0 for i in q2[0]]
    sum_2 = 0
    for lo in q2:
        count_l = 0
        for los in zip(li_long_2, list(lo)):
            li_long_2[count_l] = los[0] + los[1]
            count_l += 1
    for lo in li_long_2:
        sum_2 += lo**2
    sum_2 = math.sqrt(sum_2)
    li_long_2 = np.asarray(li_long_2)

    return np.dot(li_long_1, li_long_2) / (sum_1*sum_2)

def getEucDis(q1, q2):
    li_long_1 = []
    for lo in q1:
        li_long_1 += list(lo)
    li_long_2 = []
    for lo in q2:
        li_long_2 += list(lo)
    num = 0
    for i in range(len(li_long_1)):
        num += (li_long_1[i] - li_long_2[i])**2
    return math.sqrt(num)

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

def getF1_union(q1, q2):
    list = []
    for i in q1:
        if i not in list:
            list.append(i)
    for i in q2:
        if i not in list:
            list.append(i)
    return len(list)

def getF2_inter(q1, q2):
    list = []
    for i in q1:
        if i in q2:
            list.append(i)
    return len(list)

def getF3_sum(q1, q2):
    return len(q1 + q2)       

def dataSamp(frac=0.3):
    data = pd.read_csv('./data/merged_data.csv')

    test1 = data[data.is_duplicate==0].sample(frac=frac)
    test2 = data[data.is_duplicate==1].sample(frac=frac)

    train = data.drop(test1.index).drop(test2.index)
    test = pd.concat([test1, test2],ignore_index=True)

    test.to_csv('./data/test_merged_' + str(int(frac*10)) + '.csv', index = 0)
    train.to_csv('./data/train_merged_' + str(int(10 - frac*10)) + '.csv', index = 0)
    print('测试集训练集抽取完毕')

if __name__ == '__main__':
	getfeature(4, shape=50)