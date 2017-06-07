import keras
import pandas as pd
import numpy as np
import multiprocessing
from multprocess_def import testvalueCal, initData

if __name__ == '__main__':

	input_shape = 50
	input_length = 36
	batch_size = 32
	worker = 8

	multiprocessing.freeze_support()
	pool = multiprocessing.Pool()
	manager = multiprocessing.Manager()
	q = manager.Queue(batch_size*100)

	model = keras.models.load_model('./model/model_362-50.h5')
	vec_model = initData(input_shape)
	data = pd.read_csv('./data/merged_test_withfeature.csv',
						names=['test_id', 'question1_all', 
		                'question2_all', 'question1_stop', 
		                'question2_stop', 'question1', 
		                'question2', 
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

	count = 0
	temp_list = []
	result_data = pd.DataFrame()

	data_list = []

	for i in range(worker):
	    data_list.append(data.sample(frac=(1/(worker-i))))
	    data = data.drop(data_list[-1].index)

	for i in range(worker):
	    print('添加第'+str(i)+'个进程')
	    pool.apply_async(testvalueCal, args=(q, data_list[i], vec_model[0], vec_model[1], vec_model[2], input_length, input_shape))
	    print('第'+str(i)+'个进程添加完毕')
	pool.close()



	while 1:
		count += 1
		if count == 2345797:
			tempdata = {'q_vec_1_input': np.asarray([i[0] for i in temp_list]), 'q_vec_2_input': np.asarray([i[1] for i in temp_list]),
						'q_vec_1_all_input': np.asarray([i[2] for i in temp_list]), 'q_vec_2_all_input': np.asarray([i[3] for i in temp_list]),
						'q_vec_1_stop_input': np.asarray([i[4] for i in temp_list]), 'q_vec_2_stop_input': np.asarray([i[5] for i in temp_list]),
						'input_feature': np.asarray([i[6:-1] for i in temp_list])}
			output = model.predict(tempdata, batch_size=(2345796%batch_size))
			output = [round(i[0]) for i in output.tolist()]
			test_id = [int(i[-1]) for i in temp_list]
			result = []
			for k in zip(test_id, output):
				result.append(k)
			result = pd.DataFrame(result, columns=['test_id', 'is_duplicate'])
			result_data = pd.concat([result, result_data])
			temp_list = []
			break

		temp_list.append(q.get(True))
		if count % 100 == 0:
			print(str(count//100) + 'h has been dealed')
		if count % batch_size == 0:
			tempdata = {'q_vec_1_input': np.asarray([i[0] for i in temp_list]), 'q_vec_2_input': np.asarray([i[1] for i in temp_list]),
						'q_vec_1_all_input': np.asarray([i[2] for i in temp_list]), 'q_vec_2_all_input': np.asarray([i[3] for i in temp_list]),
						'q_vec_1_stop_input': np.asarray([i[4] for i in temp_list]), 'q_vec_2_stop_input': np.asarray([i[5] for i in temp_list]),
						'input_feature': np.asarray([i[6:-1] for i in temp_list])}
			output = model.predict(tempdata, batch_size=batch_size)
			output = [round(i[0]) for i in output.tolist()]
			test_id = [int(i[-1]) for i in temp_list]
			result = []
			for k in zip(test_id, output):
				result.append(k)
			result = pd.DataFrame(result, columns=['test_id', 'is_duplicate'])
			result_data = pd.concat([result, result_data])
			temp_list = []

	result_data = result_data.sort_values(by = 'test_id')
	result_data.to_csv('./data/myoutput.csv',index = 0)
