#coding=utf-8
from __future__ import print_function
## original version run on tensorflow 1.10
import os
import tensorflow as tf
# from keras.backend.tensorflow_backend import set_session
from tensorflow import keras
# from keras.utils.layer_utils import count_params
from tensorflow.keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import pandas as pd

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# set_session(tf.Session(config = config))
import numpy as np
# import _pickle as cp
import _pickle as cp

import random
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, concatenate
from tensorflow.keras.layers import Conv2D, GlobalMaxPooling2D
from tensorflow.python.keras.utils import np_utils
import reservoir
import time
import argparse
# from keras.utils import multi_gpu_model
from tensorflow.keras import backend as K

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
# tf.config.experimental.set_virtual_device_configuration(
#     gpus[0],
#     [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)]
# )


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


class ESAMN(object):
    def __init__(self, filepath = './RF/LP4/LP4.p', num_channels = 4, IS = 0.1, SR = 0.1, \
        multiscale = 3, n_times_res = 30, n_times_filter = 2, Dropout = 0, \
        epoch = 100, num_fold = 10):
        """
        def __init__(self, filepath = './RF/LP4/LP4.p', num_channels = 4, IS = 0.1, SR = 0.1, \
        multiscale = 3, n_times_res = 30, n_times_filter = 2, Dropout = 0, \
        epoch = 100, num_fold = 10):
        """
        self.filepath = os.path.abspath(os.path.join(os.getcwd(), filepath))
        self.num_channels = num_channels
        self.IS = IS 
        self.SR = SR
        self.multiscale = multiscale
        self.n_times_res = n_times_res
        self.n_times_filter = n_times_filter
        self.epoch = epoch
        self.num_fold = num_fold
        self.Dropout = Dropout

    def transfer_labels(self, labels):
        """
        标签转化为 one-hot
        """
        indexes = np.unique(labels)
        num_classes = indexes.shape[0]
        num_samples = labels.shape[0]
        for i in range(num_samples):
            new_label = np.argwhere(indexes == labels[i])[0][0]
            labels[i] = new_label
        labels = np_utils.to_categorical(labels, num_classes)
        return labels, num_classes

    def train(self): 
        start = time.time()
        print('Loading data...')	
        filepath = self.filepath

        skeletons, labels, _ = cp.load(open(filepath, 'rb'),encoding='iso-8859-1')
        print('Transfering labels...')
        labels, num_classes = self.transfer_labels(labels)

        num_samples, time_length, n_in = skeletons.shape
        num_folds = self.num_fold

        n_res = n_in * self.n_times_res
        IS = self.IS
        SR = self.SR
        num_channels = self.num_channels
        sparsity = [0.1 + (0.9-0.1)*(i)/(num_channels-1) for i in range(num_channels)]
        leakyrate = 1.0

        esns = [reservoir.reservoir_layer(n_in, n_res, IS, SR, sparsity[i], leakyrate) for i in range(num_channels)]

        echo_states = np.empty((num_samples, num_channels, time_length, n_res), np.float32)

        esn_start = time.time()
        print('Getting echo states...')
        for i in range(num_channels):
            echo_states[:,i,:,:] = esns[i].get_echo_states(skeletons)
        esn_end = time.time()
        
        li = list(range(num_samples))
        random.shuffle(li)

        folds = []
        num_samples_per_fold = int(num_samples / num_folds)
        p = 0
        for f in range(num_folds - 1):
            folds.append(li[p:p+num_samples_per_fold])
            p += num_samples_per_fold
        folds.append(li[p:])

        input_shape = (1, time_length, n_res)

        nb_filter = num_classes * self.n_times_filter
        nb_row = [2**i for i in range(num_channels)]
        nb_col = n_res
        kernel_initializer = 'lecun_uniform'
        activation = 'relu'
        padding = 'same' #'valid'
        strides = (1, 1)
        data_format = 'channels_first'

        optimizer = 'adam'
        loss_type = ['binary_crossentropy', 'categorical_crossentropy']
        batch_size = 4  ## hhh
        nb_epoch = self.epoch
        verbose = 2

        accs = []
        f1s = []
        trainable_count = 0
        for f in range(num_folds):

            inputs = []
            multi_pools = []
            for i in range(num_channels):
            
                input = Input(shape = input_shape)
                inputs.append(input)
            
                pools = []
                for j in range(self.multiscale):
                    conv = Conv2D(nb_filter, (nb_row[i] * (j+1), nb_col), kernel_initializer = kernel_initializer,
                                  activation = activation, padding = padding, strides = strides,
                                  data_format = data_format)(input)
                    pool = GlobalMaxPooling2D(data_format = data_format)(conv)
                    pools.append(pool)
            
                multi_pools.append(Dense(nb_filter, kernel_initializer = kernel_initializer, activation = activation)(concatenate(pools)))
            
            #features = Dense(nb_filter * num_channels / 2, kernel_initializer = kernel_initializer, activation = activation)(concatenate(multi_pools))
            features = Dense(nb_filter, kernel_initializer = kernel_initializer, activation = activation)(concatenate(multi_pools))
            if self.Dropout:
                features = Dropout(self.Dropout)(features)
            
            outputs = Dense(num_classes, kernel_initializer = kernel_initializer, activation = 'softmax')(features)
            model = Model(inputs = inputs, outputs = outputs)

            # model = multi_gpu_model(model, 2)
            trainable_count = np.sum([np.prod(v.get_shape().as_list()) for v in tf.compat.v1.trainable_variables()])
            print("需要训练优化的参数量",trainable_count)
            print(model.summary())


            model.compile(optimizer = optimizer, loss = loss_type[1], metrics = ['accuracy',
                                                                            f1_m,precision_m, recall_m])

            num_samples_test = len(folds[f])
            num_samples_train = num_samples - num_samples_test

            echo_states_test = np.empty((num_samples_test, num_channels, time_length, n_res), np.float32)
            labels_test = np.empty((num_samples_test, num_classes))
            for i in range(num_samples_test):
                echo_states_test[i] = echo_states[folds[f][i]]
                labels_test[i] = labels[folds[f][i]]
            
            echo_states_train = np.empty((num_samples_train, num_channels, time_length, n_res), np.float32)
            labels_train = np.empty((num_samples_train, num_classes))
            p = 0
            for i in range(num_folds):
                if i != f:
                    for j in range(len(folds[i])):
                        echo_states_train[p] = echo_states[folds[i][j]]
                        labels_train[p] = labels[folds[i][j]]
                        p += 1
            
            echo_states_train = [echo_states_train[:,i:i+1,:,:] for i in range(num_channels)]
            echo_states_test = [echo_states_test[:,i:i+1,:,:] for i in range(num_channels)]

            history = model.fit(echo_states_train, labels_train, batch_size = batch_size, epochs = self.epoch,
                                verbose = verbose, validation_data = (echo_states_test, labels_test))
            loss, accuracy, f1_score, precision, recall = model.evaluate(echo_states_test, labels_test, verbose=2)
            """
            inference time in the testing stage
            """
            conv_start = time.time()
            model.predict(echo_states_train)
            model.predict(echo_states_test)
            conv_end = time.time()

            accs.append(accuracy)
            f1s.append(f1_score)
            
            print('Fold :', f)
            print('Accuracy :', accs[f])


        result = dict()
        end = time.time()
        TrainingTime = end - start
        m, s = divmod(TrainingTime, 60)
        h, m = divmod(m, 60)
        deltatime = "%d:%d:%d" % (h, m, s)
        print(end - start)

        result['mean_acc'] = np.mean(accs)
        result['mean_acc_std'] = np.std(accs)
        result['mean_f1'] = np.mean(f1s)
        result['mean_f1_std'] = np.std(f1s)
        result['time'] = deltatime
        result['param_count'] = trainable_count
        #print('Mean accuracy :', sum(accs)/num_folds)
        return self.filepath, trainable_count, TrainingTime, result

def main(args):
    """
    fpath, num_channels, IS, SR, multiscale, n_times_res, 
        n_times_filter, dropout, epoch, numFold
    """
    # datasets = [['./ASD/ASD.p', 4, 0.1, 0.1, 3, 3, 2, 0, 30, 10], #30
    # ['./AL/AL.p', 4, 0.1, 0.1, 4, 3, 2, 0, 100, 10], #100
    # ['./BCI/BCI.p', 4, 0.001, 0.1, 2, 3, 20, 0, 100, 10],
    # ['./OHC/OHC.p', 5, 0.1, 0.5, 5, 30, 2, 0.2, 100, 10],
    # ['./CMU/CMUS16.p', 3, 0.01, 0.1, 3, 3, 2, 0, 100, 10], #####
    # ['./ECG/ECG.p', 4, 0.01, 0.1, 3, 30, 2, 0, 100, 10],
    # ['./Graz/Graz.p', 4, 0.1, 0.1, 3, 30, 20, 0.2, 50, 10],
    # ['./JV/JV.p', 4, 1, 0.1, 3, 3, 2, 0, 100, 10],
    # ['./Libras/Libras.p', 3, 1, 0.5, 3, 30, 2, 0, 300, 10],
    # ['./NIF_ECG/ECG.p', 5, 0.1, 0.9, 4, 30, 2, 0, 300, 10], #####
    # ['./PD/PD.p', 3, 0.01, 0.5, 2, 3, 2, 0, 150, 10],
    # ['./RF/LP1/LP1.p', 2, 0.001, 0.5, 3, 30, 2, 0.4, 100, 5],
    # ['./RF/LP2/LP2.p', 3, 0.01, 0.9, 4, 30, 2, 0.2, 100, 5],
    # ['./RF/LP3/LP3.p', 2, 0.01, 0.5, 3, 30, 2, 0, 100, 5],
    # ['./RF/LP4/LP4.p', 2, 0.001, 0.9, 3, 30, 2, 0, 100, 5], #####
    # ['./RF/LP5/LP5.p', 2, 0.01, 0.9, 3, 30, 2, 0, 100, 5],
    # ['./UWGL/UWGL.p', 5, 0.1, 0.1, 5, 30, 2, 0, 100, 10],
    # ['./Wafer/Wafer.p', 5, 0.01, 0.5, 4, 30, 20, 0, 100, 10]]

    datasets = [['./RF/LP4/LP4.p', 2, 0.001, 0.9, 3, 30, 2, 0, 100, 5]]

                lengthPara = len(datasets[0])
    for dataset in datasets:
        if len(dataset) != lengthPara:
            raise ValueError('the number of model parameters is wrong!')

    count = 0
    results = []
    for fpath, num_channels, IS, SR, multiscale, n_times_res, \
        n_times_filter, dropout, epoch, numFold in datasets[args.start_ds:args.end_ds]:
        print(fpath)
        datasetname = fpath.split('/')[1]
        filepath, trainable_count, TrainingTime, result = ESAMN(fpath,num_channels, IS, SR, multiscale,
                                                                n_times_res, n_times_filter,
                                                                dropout, epoch, numFold).train()
        print('\nresult\n', result)
        result['dataset'] = datasetname
        results.append(result)
        pd.DataFrame(results).to_csv(os.path.join(os.getcwd(),'results',args.outfile))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ESN')
    parser.add_argument('--dataset', type=str, default='LP5')
    parser.add_argument('--cuda_device', type=str, default=0, help='choose the cuda devcie')
    parser.add_argument('--start_ds', type=int, default=0, help='')
    parser.add_argument('--end_ds', type=int, default=18, help='')
    parser.add_argument('--outfile',type=str,default='ESAMN_out.csv',help='filename')

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(args.cuda_device)

    main(args)