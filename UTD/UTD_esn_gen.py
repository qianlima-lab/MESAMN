# -*- coding: UTF-8
from __future__ import print_function
from datetime import datetime
import time
import multiprocessing
from multiprocessing import Process

import os
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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
skeleton_train, labels_train = cp.load(open(filepath_train, 'rb'), encoding='iso-8859-1')
skeleton_test, labels_test = cp.load(open(filepath_test, 'rb'), encoding='iso-8859-1')

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
		if not os.path.exists('./revised/'):
			os.makedirs('./revised/')
		if not os.path.exists('./revised/UTD_esn_Q='+str(Q)+ '_SR=' + str(SR) + '_N_res_times=' + str(N_res_times) +'.npy'):
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
			cp.dump([echo_states_train, echo_states_test], open('./revised/UTD_esn_Q='+str(Q)+ '_SR=' + str(SR) + '_N_res_times=' + str(N_res_times) +'.npy', 'wb'))
		else:
			echo_states_train, echo_states_test = np.load('./revised/UTD_esn_Q='+str(Q)+ '_SR=' + str(SR) + '_N_res_times=' + str(N_res_times) +'.npy')


if __name__ == '__main__':
	for Q in [3,4,5]:
		for SR in [0.1,0.5,0.9]:
			for N_res_times in [3,30]:
				if Q == 3 and not (SR == 0.9 and N_res_times == 30):
					continue
				model = ESAMN()
				print('ok')
				p = Process(target=model.run, args=(Q, SR, N_res_times))
    			        p.start()
				# model.run(Q = Q, SR = SR, N_res_times = N_res_times)
	p.join()


