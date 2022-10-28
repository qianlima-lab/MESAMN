#coding=utf-8
from __future__ import print_function
## original version run on tensorflow 1.10
import os
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(7)
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
tf.config.experimental.set_virtual_device_configuration(
    gpus[0],
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)]
)


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
import time
import argparse
# from keras.utils import multi_gpu_model
from tensorflow.keras import backend as K

import reservoir
import utils

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

print('Loading data...')
filepath_train = './US_UTD-MHAD_train_DataSet.p'
filepath_test = './US_UTD-MHAD_test_DataSet.p'
filepath_train = os.path.abspath(os.path.join(os.getcwd(), 'UTD', filepath_train))
filepath_test = os.path.abspath(os.path.join(os.getcwd(), 'UTD', filepath_test))

skeleton_train, labels_train = cp.load(open(filepath_train, 'rb'),encoding='iso-8859-1')
skeleton_test, labels_test = cp.load(open(filepath_test, 'rb'),encoding='iso-8859-1')

# skeleton_train, labels_train = skeleton_train[:10], labels_train[:10]
# skeleton_test, labels_test = skeleton_test[:10], labels_test[:10]

print('Transfering labels...')
labels_train, labels_test, num_classes = utils.transfer_labels(labels_train, labels_test)

num_samples_train, time_length, n_in = skeleton_train.shape
num_samples_test = skeleton_test.shape[0]

n_res = n_in * 3
IS = 0.1
SR = 0.1
# sparsity = 0.5
num_channels = 4
sparsity = [0.1+i*0.8/(num_channels-1) for i in range(num_channels)]
leakyrate = 1.0

sum_acc = 0

accs = []
f1s = []
start = time.time()

for hds in range(5):

	esns = [reservoir.reservoir_layer(n_in, n_res, IS, SR, sparsity[i], leakyrate) for i in range(num_channels)]
	# esn = reservoir.reservoir_layer(n_in, n_res, IS, SR, sparsity, leakyrate)

	echo_states_train = np.empty((num_samples_train, num_channels, time_length, n_res), np.float32)
	echo_states_test = np.empty((num_samples_test, num_channels, time_length, n_res), np.float32)

	print('Getting echo states...')
	for i in range(num_channels):
		echo_states_train[:, i, :, :] = esns[i].get_echo_states(skeleton_train, 1)
		echo_states_test[:, i, :, :] = esns[i].get_echo_states(skeleton_test, 1)

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

	batch_size = 8
	nb_epoch = 100
	verbose = 2

	#inputs = Input(shape = input_shape)
	inputs = []
	multi_pools = []
	for i in range(num_channels):

		input = Input(shape = input_shape)
		inputs.append(input)

		pools = []
		for j in range(3):
			conv = Conv2D(nb_filter, (nb_row[i] * (j+1), nb_col), kernel_initializer = kernel_initializer, activation = activation, padding = padding, strides = strides, data_format = data_format)(input)
			pool = GlobalMaxPooling2D(data_format = data_format)(conv)
			pools.append(pool)
			
		multi_pools.append(Dense(nb_filter, kernel_initializer = kernel_initializer, activation = activation)(concatenate(pools)))

	#features = Dense(nb_filter * num_channels / 2, kernel_initializer = kernel_initializer, activation = activation)(concatenate(multi_pools))
	features = Dense(nb_filter, kernel_initializer = kernel_initializer, activation = activation)(concatenate(multi_pools))
	#features = Dense(nb_filter, kernel_initializer = kernel_initializer, activation = activation)(multi_pools[0])
	#features = Dropout(0.5)(features)

	outputs = Dense(num_classes, kernel_initializer = kernel_initializer, activation = 'softmax')(features)
	model = Model(inputs = inputs, outputs = outputs)
	#model.summary()

	model.compile(optimizer = optimizer, loss = loss_type[1], metrics = ['accuracy',
																		 f1_m,precision_m, recall_m])

	echo_states_train = [echo_states_train[:,i:i+1,:,:] for i in range(num_channels)]
	echo_states_test = [echo_states_test[:,i:i+1,:,:] for i in range(num_channels)]

	history = model.fit(echo_states_train, labels_train, batch_size = batch_size, epochs = nb_epoch, verbose = verbose, validation_data = (echo_states_test, labels_test))

	# if history.history['val_acc'][-1] > 0.95:
	# 	cp.dump(history.history, open('./visual/history_convesn' + str(history.history['val_acc'][-1]) + '.p','wb'))
	# 	emmm = model.predict(echo_states_test)
	# 	cp.dump(emmm, open('./visual/predict_convesn' + str(history.history['val_acc'][-1]) + '.p','wb'))
	"""
	start_time = datetime.now()
	history = model.fit(echo_states_train, labels_train, batch_size = batch_size, epochs = nb_epoch, verbose = verbose, validation_data = (echo_states_test, labels_test))
	#history = model.fit(echo_states_train, labels_train, batch_size = batch_size, epochs = nb_epoch, verbose = verbose)
	end_time = datetime.now()
	decoder_time = (end_time - start_time).seconds
	print('Decoder training time cost:', decoder_time)
	"""

	epoch = 0
	count = 0
	acc = history.history['val_accuracy'][-1]
	ff1 = history.history['val_f1_m'][-1]

	accs.append(acc)
	f1s.append(ff1)

	print('Epoch :', epoch)

result = dict()
end = time.time()
TrainingTime = end - start
m, s = divmod(TrainingTime, 60)
h, m = divmod(m, 60)
deltatime = "%d:%d:%d" % (h, m, s)
print(end - start)
print('\naccs\n:', accs)
print('\nf1s\n:', f1s)
result['mean_acc'] = np.mean(accs)
result['mean_acc_std'] = np.std(accs)
result['mean_f1'] = np.mean(f1s)
result['mean_f1_std'] = np.std(f1s)
result['time'] = deltatime
result['dataset'] = 'UTD'
print('\nresult:\n',result)
pd.DataFrame([result]).to_csv(os.path.abspath(os.path.join(os.getcwd(),'..',
										 'results', 'ESAMN_ds_UTD.csv')))

# nohup python ./UTD.py >UTD.out 2>&1 &