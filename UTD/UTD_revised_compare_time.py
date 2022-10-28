# -*- coding: UTF-8
from __future__ import print_function
from datetime import datetime

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

import reservoir
import utils

print('Loading data...')	
filepath_train = './US_UTD-MHAD_train_DataSet.p'
filepath_test = './US_UTD-MHAD_test_DataSet.p'
skeleton_train, labels_train = cp.load(open(filepath_train, 'rb'))
skeleton_test, labels_test = cp.load(open(filepath_test, 'rb'))

print('Transfering labels...')
labels_train, labels_test, num_classes = utils.transfer_labels(labels_train, labels_test)

num_samples_train, time_length, n_in = skeleton_train.shape
num_samples_test = skeleton_test.shape[0]

def print_log(s):
	with open('./revised/result.txt', 'a') as f:
		print(s, end="\n", file=f)

class ESAMN:
	def __init__(self):
		pass 
	def run(self, Q = 4, SR = 0.1, N_res_times = 3, dropout = 0):
		## 暂且忽略filter = 2, nb_multi_scale = 3, dropout = 0
		## batch size
		n_res = n_in * N_res_times
		IS = 0.1
		SR = SR
		# sparsity = 0.5
		
		dilations = [1] * Q
		#dilations = [1, 1, 1, 1]   #convmesn(skip:1,2,4,8)
		num_channels = len(dilations)
		sparsity = [0.1+i*0.8/(num_channels-1) for i in range(num_channels)]
		leakyrate = 1.0
		sum_acc = 0

		esns = [reservoir.reservoir_layer(n_in, n_res, IS, SR, sparsity[i], leakyrate) for i in range(num_channels)]
		#esn = reservoir.reservoir_layer(n_in, n_res, IS, SR, sparsity, leakyrate)

		echo_states_train = np.empty((num_samples_train, num_channels, time_length, n_res), np.float32)
		echo_states_test = np.empty((num_samples_test, num_channels, time_length, n_res), np.float32)

		print('Getting echo states...')
		for i in range(num_channels):
			#echo_states_train[:,i,:,:] = esn.get_echo_states(skeleton_train, dilations[i])
			echo_states_train[:,i,:,:] = esns[i].get_echo_states(skeleton_train, dilations[i])
			#echo_states_test[:,i,:,:] = esn.get_echo_states(skeleton_test, dilations[i])
			echo_states_test[:,i,:,:] = esns[i].get_echo_states(skeleton_test, dilations[i])


		res = []
		max_res = []
		for hds in range(5):
			input_shape = (1, time_length, n_res)
			#input_shape = (num_channels, time_length, n_res)

			nb_filter = num_classes * 2
			# nb_row = [1, 2, 4, 8]
			nb_row = [2**(i) for i in range(Q)]
			nb_col = n_res
			kernel_initializer = 'lecun_uniform'
			activation = 'relu'
			padding = 'valid'
			strides = (1, 1)
			data_format = 'channels_first'

			optimizer = 'adam'
			loss = ['binary_crossentropy', 'categorical_crossentropy']

			batch_size = 8
			nb_epoch = 100
			# nb_epoch = 2
			verbose = 1

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

			model.compile(optimizer = optimizer, loss = loss[1], metrics = ['accuracy'])

			_echo_states_train = [echo_states_train[:,i:i+1,:,:] for i in range(num_channels)]
			_echo_states_test = [echo_states_test[:,i:i+1,:,:] for i in range(num_channels)]

			history = model.fit(_echo_states_train, labels_train, batch_size = batch_size, epochs = nb_epoch, verbose = verbose, validation_data = (_echo_states_test, labels_test))

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
			max_acc = 0
			for acc in history.history['val_acc']:
				count = count + 1
				if acc > max_acc:
					max_acc = acc
					epoch = count
			res.append(history.history['val_acc'][-1])
			max_res.append(max_acc)
			sum_acc = sum_acc + history.history['val_acc'][-1]
		
			print('Epoch :', epoch)
			print('Max accuracy :', max_acc)
		average = sum_acc/5
		print("average:" + str(average))
		print_log('Q='+str(Q)+ '_SR=' + str(SR) + '_N_res_times=' + str(N_res_times))
		print_log(str(res) + '_ave = ' + str(average))
		print_log(str(max_res))

if __name__ == '__main__':
	for Q in [4]:
		for SR in [0.1]:
			for N_res_times in [3]:
				try:
					model = ESAMN()
					model.run(Q = Q, SR = SR, N_res_times = N_res_times)
				except:
					continue


