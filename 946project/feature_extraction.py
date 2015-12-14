#!/opt/sharcnet/python/2.7.5/gcc/bin/python
import sys
import os
import time
import numpy as np
import theano
import theano.tensor as T
import lasagne
import scipy.io as sio

Conv2DLayer = lasagne.layers.Conv2DLayer
dimension=300
gain=1
bias=0


input_var = T.tensor4('inputs')
num_4xd=72
num_3xd=72
num_5xd=72
num_7xd=48
num_11xd=24
    
l_in = lasagne.layers.InputLayer(shape=(1, 1, 500, 300),
                                    input_var=input_var)
out_layers = []

l_4xd = Conv2DLayer(l_in, num_filters=num_4xd, filter_size=(4, dimension), W=lasagne.init.Orthogonal(gain), b=lasagne.init.Constant(bias))
   
m_4xd = lasagne.layers.MaxPool2DLayer(l_4xd, pool_size=(497, 1))

out_layers.append(m_4xd)

l_3xd = Conv2DLayer(l_in, num_filters=num_3xd, filter_size=(3, dimension), W=lasagne.init.Orthogonal(gain), b=lasagne.init.Constant(bias))
    
m_3xd = lasagne.layers.MaxPool2DLayer(l_3xd, pool_size=(498, 1))

out_layers.append(m_3xd)

l_5xd = Conv2DLayer(l_in, num_filters=num_5xd, filter_size=(5, dimension), W=lasagne.init.Orthogonal(gain), b=lasagne.init.Constant(bias))
   
m_5xd = lasagne.layers.MaxPool2DLayer(l_5xd, pool_size=(496, 1)) 

out_layers.append(m_5xd)

l_7xd = Conv2DLayer(l_in, num_filters=num_7xd, filter_size=(7, dimension), W=lasagne.init.Orthogonal(gain), b=lasagne.init.Constant(bias))
   
m_7xd = lasagne.layers.MaxPool2DLayer(l_7xd, pool_size=(494, 1)) 

out_layers.append(m_7xd)

l_11xd = Conv2DLayer(l_in, num_filters=num_11xd, filter_size=(11, dimension), W=lasagne.init.Orthogonal(gain), b=lasagne.init.Constant(bias))
   
m_11xd = lasagne.layers.MaxPool2DLayer(l_11xd, pool_size=(490, 1)) 

out_layers.append(m_11xd)


l_out = lasagne.layers.concat(out_layers)


network = lasagne.layers.DenseLayer(
        lasagne.layers.dropout(l_out, p=.5),
        num_units=256,
        nonlinearity=lasagne.nonlinearities.tanh)

network = lasagne.layers.DenseLayer(
        lasagne.layers.dropout(network, p=.5),
        num_units=2,
        nonlinearity=lasagne.nonlinearities.softmax)


with np.load('IMDBbestaccmodel_v2.npz') as f:
    param_values = [f['arr_%d' % i] for i in range(len(f.files))]
lasagne.layers.set_all_param_values(network, param_values)

a=np.load("/scratch/rqiao/IMDB.npz")
train_data=a['arr_0']

valid_data=a['arr_2']

test_data=a['arr_4']


fn = theano.function([input_var], [lasagne.layers.get_output(l_out)])

test_p=np.zeros((25000,288),dtype=np.float32)
for i in range(20000):
    test_p[i,:]=fn(train_data[i,:,:,:].reshape(1,1,500,300))[0].flatten()
for i in range(2500):
    test_p[i+20000,:]=fn(valid_data[i+20000,:,:,:].reshape(1,1,500,300))[0].flatten()
for i in range(2500):
    test_p[i+22500,:]=fn(test_data[i+22500,:,:,:].reshape(1,1,500,300))[0].flatten()



sio.savemat('extracted_featrue.mat', {'arr': test_p})





