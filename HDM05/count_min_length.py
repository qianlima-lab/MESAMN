import numpy as np

min_length = 221

data = np.load('HDM05_DataSet_concat.p')
for i in range(data[0].shape[0]):
	num_zeros = 0
	for j in range(data[0].shape[1]-1, -1, -1):
		if sum(data[0][i][j]) == 0:
			num_zeros += 1
		else:
			break
	minn = 221 - num_zeros
	if minn < min_length:
		min_length = minn

print min_length
