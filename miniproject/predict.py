#!/opt/sharcnet/python/2.7.5/gcc/bin/python

import numpy as np
import pandas as pd
import os
import sys
from PIL import Image
import theano
import theano.tensor as T
import lasagne

Conv2DLayer = lasagne.layers.Conv2DLayer

def inception_module(l_in, num_1x1, reduce_3x3, num_3x3, reduce_5x5, num_5x5, gain=1.0, bias=0):
    """
    inception module (without the 3x3s1 pooling and projection because that's difficult in Theano right now)
    """
    out_layers = []

    # 1x1
    if num_1x1 > 0:
        l_1x1 = lasagne.layers.NINLayer(l_in, num_units=num_1x1, W=lasagne.init.Orthogonal(gain), b=lasagne.init.Constant(bias))
        out_layers.append(l_1x1)
    
    # 3x3
    if num_3x3 > 0:
        if reduce_3x3 > 0:
            l_reduce_3x3 = lasagne.layers.NINLayer(l_in, num_units=reduce_3x3, W=lasagne.init.Orthogonal(gain), b=lasagne.init.Constant(bias))
        else:
            l_reduce_3x3 = l_in
        l_3x3 = Conv2DLayer(l_reduce_3x3, num_filters=num_3x3, filter_size=(3, 3), pad="same", W=lasagne.init.Orthogonal(gain), b=lasagne.init.Constant(bias))
        out_layers.append(l_3x3)
    
    # 5x5
    if num_5x5 > 0:
        if reduce_5x5 > 0:
            l_reduce_5x5 = lasagne.layers.NINLayer(l_in, num_units=reduce_5x5, W=lasagne.init.Orthogonal(gain), b=lasagne.init.Constant(bias))
        else:
            l_reduce_5x5 = l_in
        l_5x5 = Conv2DLayer(l_reduce_5x5, num_filters=num_5x5, filter_size=(5, 5), pad="same", W=lasagne.init.Orthogonal(gain), b=lasagne.init.Constant(bias))
        out_layers.append(l_5x5)
    
    # stack
    l_out = lasagne.layers.concat(out_layers)
    return l_out


def build_cnn(input_var=None):
    
    network = lasagne.layers.InputLayer(shape=(1, 3, 256, 256),
                                        input_var=input_var)
    
    network = inception_module(
            network, num_1x1=32, reduce_3x3=48, num_3x3=64, reduce_5x5=16, num_5x5=16,)
    
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    network = inception_module(
            network, num_1x1=64, reduce_3x3=64, num_3x3=96, reduce_5x5=16, num_5x5=48,)
    
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    network = inception_module(
            network, num_1x1=96, reduce_3x3=48, num_3x3=104, reduce_5x5=16, num_5x5=48,)
    
    network = inception_module(
            network, num_1x1=80, reduce_3x3=56, num_3x3=112, reduce_5x5=24, num_5x5=64,)

    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    network = inception_module(
            network, num_1x1=192, reduce_3x3=96, num_3x3=208, reduce_5x5=16, num_5x5=48,)
    
    network = inception_module(
            network, num_1x1=160, reduce_3x3=112, num_3x3=224, reduce_5x5=24, num_5x5=64,)

    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    network = inception_module(
            network, num_1x1=156, reduce_3x3=160, num_3x3=320, reduce_5x5=32, num_5x5=128,)
    
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))


    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=256,
            nonlinearity=lasagne.nonlinearities.rectify)

    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=344,
            nonlinearity=lasagne.nonlinearities.softmax)

    return network


input_var = T.tensor4('inputs')
network = build_cnn(input_var)
with np.load('best_classification_model_omit5.npz') as f:
    param_values = [f['arr_%d' % i] for i in range(len(f.files))]
lasagne.layers.set_all_param_values(network, param_values)

prediction = lasagne.layers.get_output(network)

pred_fn = theano.function([input_var],prediction)

name=pd.read_csv('sample_submission.csv',usecols=[0])

def get_img(iname):
    temp=Image.open(iname)
    a=np.array(temp,dtype=np.float32)
    a=np.rollaxis(a,2)
    a=a.reshape(1,3,256,256)
    return(a)

os.chdir('/scratch/rqiao/Fin/Resize')

test_p=[]
for i in name['Image']:
    a=get_img(i)
    test_p.append(pred_fn(a))

print(test_p[:10])
np.savez('/home/rqiao/backup/whale_prediction344.npz',test_p)





