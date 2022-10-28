from __future__ import print_function

import os
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.1
set_session(tf.Session(config = config))

import numpy as np
import cPickle as cp

from keras.models import Model
from keras.layers import Input, Dense, Dropout, concatenate
from keras.layers import Conv2D, GlobalMaxPooling2D

import reservoir
import utils

print('Loading data...')
filepath_train = './UTD/US_UTD-MHAD_train_DataSet.p'
filepath_test = './UTD/US_UTD-MHAD_test_DataSet.p'
skeleton_train, labels_train = cp.load(open(filepath_train, 'rb'))
skeleton_test, labels_test = cp.load(open(filepath_test, 'rb'))

print('Transfering labels...')
labels_train, labels_test, num_classes = utils.transfer_labels(labels_train, labels_test)

num_samples_train, time_length, n_in = skeleton_train.shape
num_samples_test = skeleton_test.shape[0]

n_res = n_in * 3
IS = 0.1
SR = 0.1
sparsity = 0.5
leakyrate = 1.0

esn = reservoir.reservoir_layer(n_in, n_res, IS, SR, sparsity, leakyrate)

echo_states_train = np.empty((num_samples_train, 1, time_length, n_res), np.float32)
echo_states_test = np.empty((num_samples_test, 1, time_length, n_res), np.float32)

print('Getting echo states...')
echo_states_train[:,0,:,:] = esn.get_echo_states(skeleton_train, 1)
echo_states_test[:,0,:,:] = esn.get_echo_states(skeleton_test, 1)

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

batch_size = 8
nb_epoch = 100
verbose = 1

inputs = Input(shape = input_shape)
pools = []
for i in range(len(nb_row)):

	conv = Conv2D(nb_filter, (nb_row[i], nb_col), kernel_initializer = kernel_initializer, activation = activation, padding = padding, strides = strides, data_format = data_format)(inputs)
	pool = GlobalMaxPooling2D(data_format = data_format)(conv)
	pools.append(pool)

#features = Dense(nb_filter * num_channels / 2, kernel_initializer = kernel_initializer, activation = activation)(concatenate(pools))
features = Dense(nb_filter, kernel_initializer = kernel_initializer, activation = activation)(concatenate(pools))
#features = Dropout(0.5)(features)

outputs = Dense(num_classes, kernel_initializer = kernel_initializer, activation = 'softmax')(features)
model = Model(inputs = inputs, outputs = outputs)
#model.summary()

model.compile(optimizer = optimizer, loss = loss[1], metrics = ['accuracy'])

history = model.fit(echo_states_train, labels_train, batch_size = batch_size, epochs = nb_epoch, verbose = verbose, validation_data = (echo_states_test, labels_test))

epoch = 0
count = 0
acc = history.history['val_acc'][-1]

print('Epoch :', epoch)
print('Acc:', acc)