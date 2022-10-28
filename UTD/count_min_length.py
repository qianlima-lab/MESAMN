import numpy as np

min_length = 121

data = np.load('US_UTD-MHAD_train_DataSet.p')
for i in range(data[0].shape[0]):
	num_zeros = 0
	for j in range(data[0].shape[1]-1, -1, -1):
		if sum(data[0][i][j]) == 0:
			num_zeros += 1
		else:
			break
	minn = 121 - num_zeros
	if minn < min_length:
		min_length = minn

data = np.load('US_UTD-MHAD_test_DataSet.p')
for i in range(data[0].shape[0]):
	num_zeros = 0
	for j in range(data[0].shape[1]-1, -1, -1):
		if sum(data[0][i][j]) == 0:
			num_zeros += 1
		else:
			break
	minn = 121 - num_zeros
	if minn < min_length:
		min_length = minn

print min_length
