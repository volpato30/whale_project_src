#!/opt/sharcnet/python/2.7.5/gcc/bin/python

import sys
import os
import time
import numpy as np
import theano
import theano.tensor as T
import lasagne


Conv2DLayer = lasagne.layers.Conv2DLayer
dimension=300
gain=1
bias=0

def build_cnn(input_var=None,num_2xd=50,num_3xd=48,num_5xd=32,num_7xd=24):
    
    l_in = lasagne.layers.InputLayer(shape=(1, 1, 500, 300),
                                        input_var=input_var)
    out_layers = []

    l_2xd = Conv2DLayer(l_in, num_filters=num_2xd, filter_size=(2, dimension), W=lasagne.init.Orthogonal(gain), b=lasagne.init.Constant(bias))
       
    m_2xd = lasagne.layers.MaxPool2DLayer(l_2xd, pool_size=(499, 1))

    out_layers.append(m_2xd)
    
    l_3xd = Conv2DLayer(l_in, num_filters=num_3xd, filter_size=(3, dimension), W=lasagne.init.Orthogonal(gain), b=lasagne.init.Constant(bias))
        
    m_3xd = lasagne.layers.MaxPool2DLayer(l_3xd, pool_size=(498, 1))

    out_layers.append(m_3xd)

    l_5xd = Conv2DLayer(l_in, num_filters=num_5xd, filter_size=(5, dimension), W=lasagne.init.Orthogonal(gain), b=lasagne.init.Constant(bias))
       
    m_5xd = lasagne.layers.MaxPool2DLayer(l_5xd, pool_size=(496, 1)) 

    out_layers.append(m_5xd)

    l_7xd = Conv2DLayer(l_in, num_filters=num_7xd, filter_size=(7, dimension), W=lasagne.init.Orthogonal(gain), b=lasagne.init.Constant(bias))
       
    m_7xd = lasagne.layers.MaxPool2DLayer(l_7xd, pool_size=(494, 1)) 

    out_layers.append(m_7xd)


    l_out = lasagne.layers.concat(out_layers)


    network = lasagne.layers.DenseLayer(
            l_out,
            num_units=256,
            nonlinearity=lasagne.nonlinearities.tanh)

    network = lasagne.layers.DenseLayer(
            network,
            num_units=2,
            nonlinearity=lasagne.nonlinearities.softmax)

    return network


input_var = T.tensor4('inputs')
network = build_cnn(input_var)
with np.load('IMDB_model.npz') as f:
    param_values = [f['arr_%d' % i] for i in range(len(f.files))]
lasagne.layers.set_all_param_values(network, param_values)

test_data=np.load("/scratch/rqiao/946project/test_data.npz")['arr_0']

test_prediction = lasagne.layers.get_output(network, deterministic=True)

pred=T.argmax(test_prediction, axis=1)

fn = theano.function([input_var], [pred])

test_p=[]
for i in range(25000):
    test_p.append(fn(test_data[i,:,:,:].reshape(1,1,500,300)))

test_p=np.array(test_p,dtype=np.int16)
print(test_p[:100])
np.savez('prediction.npz',test_p)



