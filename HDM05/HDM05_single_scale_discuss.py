from __future__ import print_function

import os
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config = config))

import random
#random.seed(63)

import numpy as np
import cPickle as cp

from keras.models import Model
from keras.layers import Input, Dense, Dropout, concatenate
from keras.layers import Conv2D, GlobalMaxPooling2D

import reservoir
from keras.utils import np_utils

def print_log(s):
	with open('./HDM05_fiter_discuss.txt', 'a') as f:
		print(s, end="\n", file=f)


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
skeletons, labels = cp.load(open(filepath, 'rb'))

print('Transfering labels...')
labels, num_classes = transfer_labels(labels)

num_samples, time_length, n_in = skeletons.shape
n_res = n_in * 3
#IS = 1e-3
IS = 0.1
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

# nb_filter = num_classes * 2
nb_row = [5,5,5,5]
nb_col = n_res
kernel_initializer = 'lecun_uniform'
activation = 'relu'
padding = 'valid'
strides = (1, 1)
data_format = 'channels_first'

optimizer = 'adam'
loss = ['binary_crossentropy', 'categorical_crossentropy']

batch_size = 100
nb_epoch = 1000
verbose = 1

L = [x for x in range(num_samples)]
random.shuffle(L)

folds = []
p = 0
for i in range(10):
	if i == 9:
		folds.append(L[p:])
	else:
		folds.append(L[p:p+num_samples/10])
	p += num_samples / 10

L = []
for i in range(10):
	lis = []
	lis.extend(folds[i])
	for j in range(10):
		if j != i:
			lis.extend(folds[j])
	L.append(lis)

for time in [2]:
	nb_filter = num_classes * time
	max_acc = [0 for n in range(10)]
	for i in range(10):

		#inputs = Input(shape = input_shape)
		inputs = []
		multi_pools = []
		for j in range(num_channels):
		
			input = Input(shape = input_shape)
			inputs.append(input)
		
			pools = []
			## 4*5*4 = 80  VS backpone 6+12+24+48 = 90
			## another choice is 3*4*7 = 84
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
		#model.summary()
		
		model.compile(optimizer = optimizer, loss = loss[1], metrics = ['accuracy'])
		
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
		for acc in history.history['val_acc']:
			count = count + 1
			if acc > max_acc[i]:
				max_acc[i] = acc
				epoch = count
		
		print('Flod :', i)
		print('Epoch :', epoch)
		print('Max accuracy :', max_acc[i])

	print(max_acc)
	print('Ave accuracy :', sum(max_acc)/10)

	print_log([time, sum(max_acc)/10])

