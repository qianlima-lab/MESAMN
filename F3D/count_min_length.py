import numpy as np

min_length = 35

data = np.load('Florence3D_DataSet_concat.p')
for f in range(10):
	for i in range(data[f][0].shape[0]):
		num_zeros = 0
		for j in range(data[f][0].shape[1]-1, -1, -1):
			if sum(data[f][0][i][j]) == 0:
				num_zeros += 1
			else:
				break
		minn = 35 - num_zeros
		if minn < min_length:
			min_length = minn
	
print min_length
