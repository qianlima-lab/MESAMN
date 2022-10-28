from __future__ import print_function

import os
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config = config))

import numpy as np
import cPickle as cp

from keras.models import Model
from keras.layers import Input, Dense, Dropout, concatenate
from keras.layers import Conv2D, GlobalMaxPooling2D

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import *
from matplotlib.pyplot import savefig

import reservoir
from keras.utils import np_utils

def print_log(s):
	with open('./F3D_fiter_discuss.txt', 'a') as f:
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
# n_res = n_in * 3
IS = 0.1
SR = 0.1
# sparsity = 0.5
leakyrate = 1.0
flag = 1
#dilations = [1, 4, 16, 64]
# n_res = n_in * 3
# nb_filter = num_classes * 2
# dilations = [1, 1, 1, 1]
for n_res in [n_in * 3]:
	for hds_Q in [4]:
		res = []
		dilations = [1] * (hds_Q)
		num_channels = len(dilations)
		print('num_channel: ', num_channels)
		sparsity = [0.1+i*0.8/(num_channels-1) for i in range(num_channels)]

		esns = [reservoir.reservoir_layer(n_in, n_res, IS, SR, sparsity[i], leakyrate) for i in range(num_channels)]
		#esn = reservoir.reservoir_layer(n_in, n_res, IS, SR, sparsity, leakyrate)

		echo_states_temp = np.empty((num_samples, num_channels, time_length, n_res), np.float32)
		print('Getting echo states...')
		for i in range(1):
			#echo_states_temp[:,i,:,:] = esn.get_echo_states(skeletons, dilations[i])
			echo_states_temp[:,i,:,:] = esns[i].get_echo_states(skeletons, dilations[i])
		
		for i in range(1, num_channels):
			echo_states_temp[:,i,:,:] = esns[0].get_echo_states(skeletons, dilations[i])

		echo_states = []
		p = 0
		for i in range(10):
			echo_states.append(echo_states_temp[p:p+num_samples_folds[i]])
			p += num_samples_folds[i]

		for time in [2]:
			nb_filter = num_classes * (time)
			
			

			input_shape = (1, time_length, n_res)
			#input_shape = (num_channels, time_length, n_res)
			# ----------------------------------------------------------------------
			# nb_filter = num_classes * 2
			nb_row = [1, 2, 4, 8]
			nb_col = n_res



			# ----------------------------------------------------------------------
			print(nb_row)
			kernel_initializer = 'lecun_uniform'
			activation = 'relu'
			padding = 'valid'
			strides = (1, 1)
			data_format = 'channels_first'

			optimizer = 'adam'
			loss = ['binary_crossentropy', 'categorical_crossentropy']

			batch_size = 5
			nb_epoch = 300
			verbose = 1

			max_acc = [0 for n in range(10)]
			for i in range(10):

				#inputs = Input(shape = input_shape)
				inputs = []
				multi_pools = []
				for j in range(num_channels):
				
					input = Input(shape = input_shape)
					inputs.append(input)
				
					pools = []
					## from numpy import * ->会导致Bug，与keras中的concatenate冲突报错。
					for k in range(2):
						# print(nb_filter)
						conv = Conv2D(nb_filter, (nb_row[j] * (k+1), nb_col), kernel_initializer = kernel_initializer, activation = activation, padding = padding, strides = strides, data_format = data_format)(input)
						#conv = Conv2D(nb_filter, (dilations[j] * (k+1), nb_col), kernel_initializer = kernel_initializer, activation = activation, padding = padding, strides = strides, data_format = data_format)(Dropout(0.5)(input))
						pool = GlobalMaxPooling2D(data_format = data_format)(conv)
						pools.append(pool)
						# print(pools)

					#multi_pools.append(Dense(nb_filter, kernel_initializer = kernel_initializer, activation = activation)(concatenate(pools)))
					#multi_pools.append(Dense(nb_filter, kernel_initializer = kernel_initializer, activation = activation)(Dropout(0.4)(concatenate(pools))))
					# print(len(pools))
					print(pools[0].shape)
					multi_pools.append(pools[0])
					# multi_pools.append(pools)
					# multi_pools.append(Dense(nb_filter, kernel_initializer = kernel_initializer, activation = activation)(pools[0]))

				#features = Dense(nb_filter, kernel_initializer = kernel_initializer, activation = activation)(concatenate(multi_pools))
				print(multi_pools)
				# features = Dense(nb_filter, kernel_initializer = kernel_initializer, activation = activation)(Dropout(0.4)(multi_pools[0]))
				features = Dense(nb_filter, kernel_initializer = kernel_initializer, activation = activation)(Dropout(0.4)(concatenate(multi_pools)))
				
				#features = Dropout(0.5)(features)
				
				outputs = Dense(num_classes, kernel_initializer = kernel_initializer, activation = 'softmax')(features)
				model = Model(inputs = inputs, outputs = outputs)
				#model.summary()
				
				model.compile(optimizer = optimizer, loss = loss[1], metrics = ['accuracy'])
				
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
			print('Ave accuracy :', sum(max_acc)/10)
			flag +=1
			print_log([time, sum(max_acc)/10])
			# figure
			# subplot(1,2,1)
			# plt.plot(history.history[acc])
			# plt.plot(history.history[val_acc])
			# legend(['Training accuracy','Testing accuracy'])
			# xlabel('Epoch')
			# ylabel('Acc')
			# subplot(1,2,2)
			# plt.plot(history.history[loss])
			# plt.plot(history.history[val_loss])
			# legend(['Training loss','Testing loss'])
			# xlabel('Epoch')
			# ylabel('Loss')
			# savefig('./plot/F3D'+str(flag)+"N="+str(n_res)+'Q='+str(hds_Q+1)+"K="+str(filter_times)+'.jpg',dpi = 2000)
