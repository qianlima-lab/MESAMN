import numpy as np
import _pickle as cp

# data_name = 'Florence3D_DataSet.p'
# data_temp = cp.load(open(data_name, 'rb'), encoding='iso-8859-1')
# data = []
# for i in range(len(data_temp)):
# 	left_hand, right_hand, left_leg, right_leg, central_trunk, labels = data_temp[i]
# 	body = np.concatenate((left_hand, right_hand, left_leg, right_leg, central_trunk), axis = 2)
# 	print(body.shape)
# 	data.append([body, labels])
# cp.dump(data, open('Florence3D_DataSet_concat.p', "wb"))

def create_N_C_T_V_M(data):
    N, T, VC = data.shape
    data = data.reshape((N, T, 3, 3, 1)) # n t v c m
    return data

data_name = 'Florence3D_DataSet.p'
data_temp = cp.load(open(data_name, 'rb'), encoding='iso-8859-1')
data = []
for i in range(len(data_temp)):
	left_hand, right_hand, left_leg, right_leg, central_trunk, labels = data_temp[i]
	left_hand = create_N_C_T_V_M(left_hand)
	right_hand = create_N_C_T_V_M(right_hand)
	left_leg = create_N_C_T_V_M(left_leg)
	right_leg = create_N_C_T_V_M(right_leg)
	central_trunk = create_N_C_T_V_M(central_trunk)
	body = np.concatenate((central_trunk, left_hand, right_hand, left_leg, right_leg), axis = 2)
	body = body.transpose(0, 3, 1, 2, 4)
	print(body.shape)
	data.append([body, labels])
cp.dump(data, open('Florence3D_DataSet_N_C_T_V_M.p', "wb"))

# a = [(0, 2), (1, 2), (2, 14), (3, 5), (4, 5), (5, 14), (6, 8), (7, 8), (8, 14), (9, 11), (10, 11), (11, 14), (12, 14),
# 	 (13, 14)]