# 模型的主程序



import keras
from keras.layers import Input, LSTM, Dense, Dropout
from keras.models import Model, Sequential
from keras.models import load_model
from keras.utils import plot_model
from keras import metrics
from multprocess_def import justDoitData, dataSamp, createlog
import datetime
from keras import regularizers
import matplotlib.pyplot as plt
import numpy as np



# from keras.utils import plot_model

if __name__ == '__main__':
    # 开始时的epochs
    trainedepochs = 352
    everyepochs = 10

    print('new!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

    # 循环计数
    loop_count = 0
    # while 1:
    loop_count += 1
    trainnote = '将训练集负样本抽样重复使得正样本比例修改为 0.17647，总样本数变为592076，测试集依然为0.144907'
    input_shape = 50
    input_length = 36
    patch_size = 128
    frac = 0.3
    train_patch = 3158
    test_patch = 945
    lstm_num = 3
    trainedepochs = trainedepochs + (loop_count-1)*everyepochs
    epochs = everyepochs

    note_model = createlog(trainedepochs, epochs, input_shape, input_length, patch_size, frac, train_patch, test_patch, lstm_num, trainnote)

    # ################## LSTM双输入 ##################
    # # 所有问题的lstm输入
    # tweet_a = Input(shape=(input_length, input_shape), name='q_vec_1_input')
    # tweet_b = Input(shape=(input_length, input_shape), name='q_vec_2_input')
    # # 所有问题去除所有词的lstm输入
    # tweet_a_all = Input(shape=(input_length, input_shape), name='q_vec_1_all_input')
    # tweet_b_all = Input(shape=(input_length, input_shape), name='q_vec_2_all_input')
    # # 所有问题去除停用词的lstm输入
    # tweet_a_stop = Input(shape=(input_length, input_shape), name='q_vec_1_stop_input')
    # tweet_b_stop = Input(shape=(input_length, input_shape), name='q_vec_2_stop_input')
    # # 定义三个lstm
    # shared_lstm = LSTM(64)
    # shared_lstm_all = LSTM(64)
    # shared_lstm_stop = LSTM(64)
    # # 所有问题输入lstm
    # encoded_a = shared_lstm(tweet_a)
    # encoded_b = shared_lstm(tweet_b)
    # # 所有问题去除所有词输入lstm
    # encoded_a_all = shared_lstm_all(tweet_a_all)
    # encoded_b_all = shared_lstm_all(tweet_b_all)
    # # 所有问题去除停用词输入lstm
    # encoded_a_stop = shared_lstm_stop(tweet_a_stop)
    # encoded_b_stop = shared_lstm_stop(tweet_b_stop)
    # # 将三个lstm各自进行结合
    # merged_vector = keras.layers.concatenate([encoded_a, encoded_b], axis=-1)
    # merged_vector_all = keras.layers.concatenate([encoded_a_all, encoded_b_all], axis=-1)
    # merged_vector_stop = keras.layers.concatenate([encoded_a_stop, encoded_b_stop], axis=-1)    
    # # lstm的结果放大进入隐藏层
    # # hidden_dropour = Dropout(0.5)(merged_vector)
    # hidden = Dense(384, activation='tanh', name='hidden_dense')(merged_vector)
    # # hidden_dropour_all = Dropout(0.5)(merged_vector_all)
    # hidden_all = Dense(384, activation='tanh', name='hidden_dense_all')(merged_vector_all)
    # # hidden_dropour_stop = Dropout(0.5)(merged_vector_stop)
    # hidden_stop = Dense(384, activation='tanh', name='hidden_dense_stop')(merged_vector_stop)
    # # lstm的结果缩小进入第二个隐藏层
    # hidden_1 = Dense(16, activation='tanh', name='hidden_1_dense')(hidden)
    # hidden_1_all = Dense(16, activation='tanh', name='hidden_1_dense_all')(hidden_all)
    # hidden_1_stop = Dense(16, activation='tanh', name='hidden_1_dense_stop')(hidden_stop)
    # # 其他所有特征输入网络
    # input_feature = Input(shape=(66,), name='input_feature')
    # # 将三个缩小后的lstm隐藏层以及其他特征结合
    # merged = keras.layers.concatenate([hidden_1, hidden_1_all, hidden_1_stop, input_feature], axis=-1)
    # # 结合后的特征经过神经网络
    # hidden_last = Dense(64, activation='tanh', name = 'hidden_last')(merged)
    # # put_dropout = Dropout(0.5)(hidden_last)
    # # put_dropout = Dropout(0.5)(merged)
    # # predictions = Dense(1, activation='sigmoid', name='output_dense', kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l1(0.01))(hidden_last)
    # predictions = Dense(1, activation='sigmoid', name='output_dense')(hidden_last)


    # model = Model(inputs=[input_feature, tweet_a, tweet_b, tweet_a_all, tweet_b_all, tweet_a_stop, tweet_b_stop], outputs=predictions)
    # model.summary()
   
    # model.compile(optimizer=keras.optimizers.SGD(lr=0.02, momentum=0.0, decay=0.0, nesterov=False), 
    #               loss='binary_crossentropy',
    #               metrics=['accuracy', metrics.precision, metrics.recall, metrics.fmeasure])

    # plot_model(model, to_file='./modelVis/model-' + str(datetime.datetime.now())[:19].replace(':','-') + '.png')

    # print('模型编译成功，开始进行数据预处理')

    ################## 获取数据并进行训练 ##################
    model = keras.models.load_model('./model/model_'+str(trainedepochs)+'-'+str(input_shape)+'.h5')
    # print(logs.keys())
    # plot_loss_callback = keras.callbacks.LambdaCallback(on_batch_begin=lambda batch, logs: plt.plot(np.arange(batch), logs['loss']))
    tenB = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=1)
    #early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto')

    # 带有 class_weight 的训练
    # model.fit_generator(justDoitData('train', frac, patch_size, input_length, input_shape, worker=16), 
    #                     train_patch, epochs=epochs, callbacks = [tenB], workers = 1, 
    #                     class_weight={1:0.63,0:0.37}, max_q_size = patch_size)
    model.fit_generator(justDoitData('train', frac, patch_size, input_length, input_shape, worker=16), 
                        train_patch, epochs=epochs, callbacks = [tenB], workers = 1, max_q_size = patch_size)

    model.save('./model/model_'+str(trainedepochs+epochs)+'-'+str(input_shape)+'.h5')
    
    # tensorboard --logdir=C:\Users\98732\Desktop\workplace\quora\logs
    output = model.evaluate_generator(justDoitData('test', frac, patch_size, input_length, input_shape, worker=16), 
                                        workers = 1, steps = test_patch,  max_q_size = patch_size)
    print('\n****************** output ******************')
    print('    loss : %s' % (output[0]))
    print('    acc : %s' % (output[1]))
    
    f = open('model_output.txt','a')
    f.write('******************************************\n')
    f.write('    ./model/model_'+str(trainedepochs+epochs)+'-'+str(input_shape)+'.h5\n')
    f.write('    metrics : \n      test loss : %s' % (output[0]) + '\n')
    f.write('      test acc : %s' % (output[1]) + '\n')
    f.write('      test precision : %s' % (output[2]) + '\n')
    f.write('      test recall : %s' % (output[3]) + '\n')
    f.write('      test fmeasure : %s' % (output[4]) + '\n')
    f.write('    parameter : \n')

    f.write(note_model)
    f.write('******************************************\n')
    f.close()


