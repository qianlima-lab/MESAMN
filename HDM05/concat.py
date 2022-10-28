import numpy as np
import _pickle as cp

# data_name = 'HDM05_DataSet.p'
# data_temp = cp.load(open(data_name, 'rb'), encoding='iso-8859-1')
# left_hand, right_hand, left_leg, right_leg, central_trunk, labels = data_temp
# body = np.concatenate((left_hand, right_hand, left_leg, right_leg, central_trunk), axis = 2)
# data = [body, labels]
# cp.dump(data, open('HDM05_DataSet_concat.p', "wb"))

def create_N_C_T_V_M(data):
    N, T, VC = data.shape
    data = data.reshape((N, T, 3, int(VC/3), 1))
    return data

data_name = 'HDM05_DataSet.p'
data_temp = cp.load(open(data_name, 'rb'), encoding='iso-8859-1')
left_hand, right_hand, left_leg, right_leg, central_trunk, labels = data_temp
left_hand = create_N_C_T_V_M(left_hand)
right_hand = create_N_C_T_V_M(right_hand)
left_leg = create_N_C_T_V_M(left_leg)
right_leg = create_N_C_T_V_M(right_leg)
central_trunk = create_N_C_T_V_M(central_trunk)

body = np.concatenate((left_hand, right_hand, left_leg, right_leg, central_trunk), axis = 3)
body = body.transpose(0, 2, 1, 3, 4)
data = [body, labels]
cp.dump(data, open('HDM05_DataSet_N_C_T_V_M_cv.p', "wb"))

a = [(i, 6) for i in range(6)] + [(i, 10) for i in range(7, 10)] + [(i, 16) for i in range(11, 17)] + \
    [(i, 21) for i in range(18, 21)] + [(i, 25) for i in range(22, 25)] + [(6, 25), (10, 25), (16, 25), (21, 25)]
lh_v = [0,1,2,3,4,5,6]
ll_v = [7,8,9,10]
rh_v = [11,12,13,14,15,16,17]
rl_v = [18,19,20,21]
ct_v = [22,23,24,25]
a = [(0,1),(1,2),(2,3),(3,4),(5,6),(0,23),(7,8),(8,9),(9,10),(7,25),(11,12),(12,13),(13,14),(14,15),
     (15,16),(16,17),(11,23),(18,19),(19,20),(20,21),(18,25),(22,23),(23,24),(24,25)]