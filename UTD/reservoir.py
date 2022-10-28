import numpy as np

import scipy as sp
from scipy.sparse import *
from scipy.sparse.linalg import *
import numpy.linalg as LA  
import matplotlib
import warnings
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def softmax(x):
    """
    Compute the softmax function for each row of the input x.

    Arguments:
    x -- A N dimensional vector or M x N dimensional numpy matrix.

    Return:
    x -- You are allowed to modify x in-place
    """
    orig_shape = x.shape

    if len(x.shape) > 1:
        # Matrix
        exp_minmax = lambda x: np.exp(x - np.max(x))
        denom = lambda x: 1.0 / np.sum(x)
        x = np.apply_along_axis(exp_minmax,1,x)
        denominator = np.apply_along_axis(denom,1,x) 
        
        if len(denominator.shape) == 1:
            denominator = denominator.reshape((denominator.shape[0],1))
        
        x = x * denominator
    else:
        # Vector
        x_max = np.max(x)
        x = x - x_max
        numerator = np.exp(x)
        denominator =  1.0 / np.sum(numerator)
        x = numerator.dot(denominator)
    
    assert x.shape == orig_shape
    return x
	
class reservoir_layer(object):
	
	def __init__(self, n_in, n_res, IS, SR, sparsity, leakyrate, use_bias = False):
		
		self.n_in = n_in
		self.n_res = n_res
		self.IS = IS
		self.SR = SR
		self.sparsity = sparsity
		self.leakyrate = leakyrate
		self.use_bias = use_bias

		self.W_in = 2 * np.random.random(size = (self.n_res, self.n_in)) - 1

		W_res_temp = sp.sparse.rand(self.n_res, self.n_res, self.sparsity)
		vals, vecs = sp.sparse.linalg.eigsh(W_res_temp, k = 1)
		self.W_res = (self.SR * W_res_temp / vals[0]).toarray()

		b_bound = 0.1
		self.b = 2 * b_bound * np.random.random(size = (self.n_res)) - b_bound
	
	def get_echo_states(self, series = [], dilation = [], self_attention_flag = 1):
		
		num_samples, time_length, _ = series.shape
		echo_states = np.empty((num_samples, time_length, self.n_res), np.float64)
		for i in range(num_samples):
			
			collect_states = np.empty((time_length, self.n_res), np.float64)
			x = [np.zeros((self.n_res)) for n in range(dilation)]

			for t in range(time_length):
				
				u = series[i, t]
				index = t % dilation

				if self.use_bias:
					xUpd = np.tanh(np.dot(self.W_in, self.IS * u) + np.dot(self.W_res, x[index]) + self.b)
				else:
					xUpd = np.tanh(np.dot(self.W_in, self.IS * u) + np.dot(self.W_res, x[index]))

				x[index] = (1 - self.leakyrate) * x[index] + self.leakyrate * xUpd

				collect_states[t] = x[index]
			#print collect_states.shape,np.linalg.matrix_rank(collect_states)
			# #self attention
			#np.savetxt('states.txt',collect_states)

			
            #self attention
			if self_attention_flag:
				temp_states = np.dot(collect_states,collect_states.T)
				norm = LA.norm(collect_states,axis=1,keepdims=True)
				temp_norm = np.dot(norm,norm.T)
				cos = np.divide(temp_states, temp_norm, where= temp_norm!=0)
				mark = np.where(cos == 0)	
				cos [ np.where( cos <0.999 ) ] = -16
				cos [ mark ] = -1024
				cos = softmax(cos)
				collect_states = np.dot(cos,collect_states)




			echo_states[i] = collect_states
		return echo_states

def visualize_echo_state(echo_state, name):

	time_length = echo_state.shape[0]
	n_res = echo_state.shape[1]
	
	plt.figure(figsize = (19.20, 10.80))

	x_labels = range(n_res)
	y_labels = range(time_length)

	x_ticks = np.array(range(len(x_labels)))
	y_ticks = np.array(range(len(y_labels)))

	plt.gca().set_xticks(x_ticks, minor = True)
	plt.gca().set_yticks(y_ticks, minor = True)
	plt.gca().xaxis.set_ticks_position('none')
	plt.gca().yaxis.set_ticks_position('none')

	plt.grid(None, which = 'minor', linestyle = 'none')

	plt.imshow(echo_state.T, interpolation = 'nearest', aspect = 'auto')

	plt.colorbar()

	plt.xlabel('Time Direction')
	plt.ylabel('Reservoir Neurons')

	plt.savefig(name)

	plt.close('all')
