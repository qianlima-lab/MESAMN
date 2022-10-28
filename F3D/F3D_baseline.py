#coding=utf-8
from __future__ import print_function
## original version run on tensorflow 1.10
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(3)

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


def transfer_labels(labels):

	indexes = np.unique(labels)
	num_classes = indexes.shape[0]
	num_samples = labels.shape[0]

	for i in range(num_samples):
		new_label = np.argwhere(indexes == labels[i])[0][0]
		labels[i] = new_label

	labels = np_utils.to_categorical(labels, num_classes)

	return labels, num_classes

L = []
L.append([[3, 4, 5, 6, 10], [1, 2, 7, 8, 9]])
L.append([[1, 3, 5, 9, 10], [2, 4, 6, 7, 8]])
L.append([[1, 2, 4, 6, 9], [3, 5, 7, 8, 10]])
L.append([[2, 3, 6, 8, 10], [1, 4, 5, 7, 9]])
L.append([[3, 4, 6, 7, 9], [1, 2, 5, 8, 10]])
L.append([[1, 2, 6, 8, 10], [3, 4, 5, 7, 9]])
L.append([[1, 2, 6, 7, 10], [3, 4, 5, 8, 9]])
L.append([[1, 3, 5, 6, 8], [2, 4, 7, 9, 10]])
L.append([[2, 3, 6, 9, 10], [1, 4, 5, 7, 8]])
L.append([[1, 2, 6, 9, 10], [3, 4, 5, 7, 8]])

print('Loading data...')	
filepath = './F3D/Florence3D_DataSet_concat.p'
filepath = os.path.abspath(os.path.join(os.getcwd(), filepath))

data = cp.load(open(filepath, 'rb'))
skeletons = []
labels = []
num_samples_folds = []
for i in range(10):
	skeleton, label = data[i]
	skeletons.append(skeleton)
	labels.append(label)
	num_samples_folds.append(label.shape[0])
skeletons = np.concatenate(skeletons, axis = 0)
labels = np.concatenate(labels, axis = 0)

print('Transfering labels...')
labels_temp, num_classes = transfer_labels(labels)

labels = []
p = 0
for i in range(10):
	labels.append(labels_temp[p:p+num_samples_folds[i]])
	p += num_samples_folds[i]

num_samples, time_length, n_in = skeletons.shape
n_res = n_in * 3
IS = 0.1
# SR = 0.9
SR = 0.1
sparsity = 0.5
leakyrate = 1.0

esn = reservoir.reservoir_layer(n_in, n_res, IS, SR, sparsity, leakyrate)

echo_states_temp = np.empty((num_samples, 1, time_length, n_res), np.float32)
print('Getting echo states...')
echo_states_temp[:,0,:,:] = esn.get_echo_states(skeletons, 1)

echo_states = []
p = 0
for i in range(10):
	echo_states.append(echo_states_temp[p:p+num_samples_folds[i]])
	p += num_samples_folds[i]

input_shape = (1, time_length, n_res)
#input_shape = (num_channels, time_length, n_res)

nb_filter = num_classes * 2
nb_row = [1, 2, 3]
nb_col = n_res
kernel_initializer = 'lecun_uniform'
activation = 'relu'
padding = 'valid'
strides = (1, 1)
data_format = 'channels_first'

optimizer = 'adam'
loss = ['binary_crossentropy', 'categorical_crossentropy']

batch_size = 5
nb_epoch = 300
verbose = 2

accs = []
f1s = []
start = time.time()

for i in range(10):

	inputs = Input(shape = input_shape)
	pools = []
	for j in range(len(nb_row)):
		conv = Conv2D(nb_filter, (nb_row[j], nb_col), kernel_initializer = kernel_initializer, activation = activation, padding = padding, strides = strides, data_format = data_format)(inputs)
		pool = GlobalMaxPooling2D(data_format = data_format)(conv)
		pools.append(pool)
	
	#features = Dense(nb_filter * num_channels / 2, kernel_initializer = kernel_initializer, activation = activation)(concatenate(pools))
	features = Dense(nb_filter, kernel_initializer = kernel_initializer, activation = activation)(Dropout(0.4)(concatenate(pools)))
	#features = Dropout(0.5)(features)

	outputs = Dense(num_classes, kernel_initializer = kernel_initializer, activation = 'softmax')(features)
	model = Model(inputs = inputs, outputs = outputs)
	#model.summary()
	
	model.compile(optimizer = optimizer, loss = loss[1], metrics = ['accuracy',
																	f1_m, precision_m, recall_m])
	
	echo_states_train_list = []
	echo_states_test_list = []
	for k in range(5):
		echo_states_train_list.append(echo_states[L[i][0][k]-1])
		echo_states_test_list.append(echo_states[L[i][1][k]-1])
	echo_states_train = np.concatenate(echo_states_train_list, axis = 0)
	echo_states_test = np.concatenate(echo_states_test_list, axis = 0)

	labels_train_list = []
	labels_test_list = []
	for k in range(5):
		labels_train_list.append(labels[L[i][0][k]-1])
		labels_test_list.append(labels[L[i][1][k]-1])
	labels_train = np.concatenate(labels_train_list, axis = 0)
	labels_test = np.concatenate(labels_test_list, axis = 0)

	labels_train = np.array(labels_train)
	labels_test = np.array(labels_test)
	print(labels_train.shape, labels_test.shape)



	#echo_states_train = [echo_states_train[:,x:x+1,:,:] for x in range(num_channels)]
	#echo_states_test = [echo_states_test[:,x:x+1,:,:] for x in range(num_channels)]
	
	history = model.fit(echo_states_train, labels_train, batch_size = batch_size, epochs = nb_epoch, verbose = verbose, validation_data = (echo_states_test, labels_test))
	
	epoch = 0
	count = 0
	acc = history.history['val_accuracy'][1]
	ff1 = history.history['val_f1_m'][-1]

	accs.append(acc)
	f1s.append(ff1)
	
	print('Flod :', i)
	print('Epoch :', epoch)


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
result['dataset'] = 'F3D'

pd.DataFrame([result]).to_csv(os.path.join(os.getcwd(),'..',
										 'results', 'ESAMN_ds_F3D.csv'))

# nohup python ./F3D_baseline.py >F3D.out 2>&1 &

