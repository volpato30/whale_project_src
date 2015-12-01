#!/opt/sharcnet/python/2.7.5/gcc/bin/python

import sys
import os
import time
import numpy as np
import theano
import theano.tensor as T
import lasagne



dimension=300

def convlayer(l_in, num_2xd, num_3xd, num_5xd, gain=1.0, bias=0):
    
    out_layers = []

    if num_2xd > 0:
        
        l_2xd = Conv2DLayer(l_in, num_filters=num_2xd, filter_size=(2, dimension), pad=(1,0), W=lasagne.init.Orthogonal(gain), b=lasagne.init.Constant(bias))
        out_layers.append(l_2xd)
    
    if num_3xd > 0:
        
        l_3xd = Conv2DLayer(l_in, num_filters=num_3xd, filter_size=(3, dimension), pad=(2,0), W=lasagne.init.Orthogonal(gain), b=lasagne.init.Constant(bias))
        out_layers.append(l_3xd)

    if num_5xd > 0:
        
        l_5xd = Conv2DLayer(l_in, num_filters=num_5xd, filter_size=(5, dimension), pad=(4,0), W=lasagne.init.Orthogonal(gain), b=lasagne.init.Constant(bias))
        out_layers.append(l_5xd)
    
    # stack
    l_out = lasagne.layers.concat(out_layers)
    return l_out


def build_cnn(input_var=None):
    
    network = lasagne.layers.InputLayer(shape=(10, 1, 2000, 300),
                                        input_var=input_var)
    
    network = convlayer(network, num_2xd=32, num_3xd=32, num_5xd=32)
    
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2000, 1))

    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=128,
            nonlinearity=lasagne.nonlinearities.rectify)

    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=2,
            nonlinearity=lasagne.nonlinearities.softmax)

    return network



input_var = T.tensor4('inputs')
network = build_cnn(input_var)
with np.load('miniproject_bestmodel.npz') as f:
    param_values = [f['arr_%d' % i] for i in range(len(f.files))]
lasagne.layers.set_all_param_values(network, param_values)

test_data=np.load("/scratch/rqiao/mptest.npz")['arr_0']

test_prediction = lasagne.layers.get_output(network, deterministic=True)

pred=T.argmax(test_prediction, axis=1)

fn = theano.function([input_var], [pred])

test_p=fn(test_data)

print(test_p[:10])
np.savez('prediction.npz',test_p)



