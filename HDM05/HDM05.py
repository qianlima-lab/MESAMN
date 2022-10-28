#coding=utf-8
from __future__ import print_function
## original version run on tensorflow 1.10
import os
import sys
sys.path.append('..')

os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(4)

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

print('Loading data...')	
filepath = './HDM05/HDM05_DataSet_concat.p'
filepath = os.path.abspath(os.path.join(os.getcwd(), filepath))
skeletons, labels = cp.load(open(filepath, 'rb'),encoding='iso-8859-1')

# skeletons = skeletons[:10]
# labels = labels[:10]

print('Transfering labels...')
labels, num_classes = transfer_labels(labels)

num_samples, time_length, n_in = skeletons.shape
n_res = n_in * 3
IS = 1e-3
# IS = 0.1
SR = 0.1
# sparsity = 0.5
leakyrate = 1.0
#dilations = [1, 4, 16, 64]
dilations = [1, 1, 1, 1]
num_channels = len(dilations)
sparsity = [0.1+i*0.8/(num_channels-1) for i in range(num_channels)]

esns = [reservoir.reservoir_layer(n_in, n_res, IS, SR, sparsity[i], leakyrate) for i in range(num_channels)]
#esn = reservoir.reservoir_layer(n_in, n_res, IS, SR, sparsity, leakyrate)

echo_states = np.empty((num_samples, num_channels, time_length, n_res), np.float32)
print('Getting echo states...')
for i in range(num_channels):
	#echo_states[:,i,:,:] = esn.get_echo_states(skeletons, dilations[i])
	echo_states[:,i,:,:] = esns[i].get_echo_states(skeletons, dilations[i])

input_shape = (1, time_length, n_res)
#input_shape = (num_channels, time_length, n_res)

nb_filter = num_classes * 2
nb_row = [1, 2, 4, 8]
nb_col = n_res
kernel_initializer = 'lecun_uniform'
activation = 'relu'
padding = 'valid'
strides = (1, 1)
data_format = 'channels_first'

optimizer = 'adam'
loss_type = ['binary_crossentropy', 'categorical_crossentropy']

batch_size = 100
nb_epoch = 100
verbose = 2

L = [x for x in range(num_samples)]
random.shuffle(L)

folds = []
p = 0
for i in range(10):
	if i == 9:
		folds.append(L[int(p):])
	else:
		folds.append(L[int(p):int(p+num_samples/10)])
	p += num_samples / 10

L = []
for i in range(10):
	lis = []
	lis.extend(folds[i])
	for j in range(10):
		if j != i:
			lis.extend(folds[j])
	L.append(lis)

accs = []
f1s = []
start = time.time()

trainable_count = 0
for i in range(10):

	#inputs = Input(shape = input_shape)
	inputs = []
	multi_pools = []
	for j in range(num_channels):
	
		input = Input(shape = input_shape)
		inputs.append(input)
	
		pools = []
		for k in range(3):
			conv = Conv2D(nb_filter, (nb_row[j] * (k+1), nb_col), kernel_initializer = kernel_initializer, activation = activation, padding = padding, strides = strides, data_format = data_format)(input)
			pool = GlobalMaxPooling2D(data_format = data_format)(conv)
			pools.append(pool)
	
		#multi_pools.append(Dense(nb_filter, kernel_initializer = kernel_initializer, activation = activation)(concatenate(pools)))
		multi_pools.append(Dense(nb_filter, kernel_initializer = kernel_initializer, activation = activation)(Dropout(0.5)(concatenate(pools))))
		#multi_pools.append(pools[0])

	#features = Dense(nb_filter, kernel_initializer = kernel_initializer, activation = activation)(concatenate(multi_pools))
	features = Dense(nb_filter, kernel_initializer = kernel_initializer, activation = activation)(Dropout(0.5)(concatenate(multi_pools)))
	#features = Dense(nb_filter, kernel_initializer = kernel_initializer, activation = activation)(Dropout(0.5)(multi_pools[0]))
	#features = Dropout(0.5)(features)
	
	outputs = Dense(num_classes, kernel_initializer = kernel_initializer, activation = 'softmax')(features)
	model = Model(inputs = inputs, outputs = outputs)
	trainable_count = np.sum([np.prod(v.get_shape().as_list()) for v in tf.compat.v1.trainable_variables()])

	#model.summary()
	
	model.compile(optimizer = optimizer, loss = loss_type[1], metrics = ['accuracy',f1_m,precision_m, recall_m])
	
	p = 0
	echo_states_test = np.empty((len(folds[i]), num_channels, time_length, n_res))
	labels_test = np.empty((len(folds[i]), num_classes))
	for n in range(len(folds[i])):
		echo_states_test[n] = echo_states[L[i][p]]
		labels_test[n] = labels[L[i][p]]
		p += 1
	
	echo_states_train = np.empty((num_samples - len(folds[i]), num_channels, time_length, n_res))
	labels_train = np.empty((num_samples - len(folds[i]), num_classes))
	for n in range(num_samples - len(folds[i])):
		echo_states_train[n] = echo_states[L[i][p]]
		labels_train[n] = labels[L[i][p]]
		p += 1

	echo_states_train = [echo_states_train[:,x:x+1,:,:] for x in range(num_channels)]
	echo_states_test = [echo_states_test[:,x:x+1,:,:] for x in range(num_channels)]
	
	history = model.fit(echo_states_train, labels_train, batch_size = batch_size, epochs = nb_epoch, verbose = verbose, validation_data = (echo_states_test, labels_test))
	
	epoch = 0
	count = 0
	acc = history.history['val_accuracy'][-1]
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
result['param_count'] = trainable_count
result['dataset'] = 'HDM05'

pd.DataFrame([result]).to_csv(os.path.join(os.getcwd(), '..', 'results', 'HDM05.csv'))


