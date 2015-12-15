#!/opt/sharcnet/python/2.7.5/gcc/bin/python

import sys
import os
import time
import numpy as np
import theano
import theano.tensor as T
import lasagne
from lasagne.regularization import regularize_layer_params, l2
from lasagne.layers import LocalResponseNormalization2DLayer as LRNLayer
from lasagne.layers import GlobalPoolLayer

BATCHSIZE=1

Conv2DLayer = lasagne.layers.Conv2DLayer

def inception_module(l_in,pool_filters, num_1x1, reduce_3x3, num_3x3, reduce_5x5, num_5x5, gain=1.0, bias=0):
    """
    inception module (without the 3x3s1 pooling and projection because that's difficult in Theano right now)
    """
    out_layers = []

    if pool_filters > 0:
        l_pool = lasagne.layers.MaxPool2DLayer(l_in, pool_size=3, stride=1, pad=1)
        l_pool_reduced = lasagne.layers.NINLayer(l_pool, num_units=pool_filters, W=lasagne.init.Orthogonal(gain), b=lasagne.init.Constant(bias))
        out_layers.append(l_pool_reduced)

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
    
    network = lasagne.layers.InputLayer(shape=(BATCHSIZE, 3, 256, 256),
                                        input_var=input_var)
    
    network = Conv2DLayer(network, num_filters=64, filter_size=7, stride=2, pad=3)

    network = lasagne.layers.MaxPool2DLayer(network,pool_size=3, stride=2, ignore_border=False)

    network= LRNLayer(network, alpha=0.00002, k=1)

    network = lasagne.layers.NINLayer(network, num_units=64, W=lasagne.init.Orthogonal(1), b=lasagne.init.Constant(0))
    network = Conv2DLayer(network, 192, 3, pad=1)
    network = LRNLayer(network, alpha=0.00002, k=1)
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=3, stride=2)

    
    network = inception_module(
            network,pool_filters=32, num_1x1=64, reduce_3x3=96, num_3x3=128, reduce_5x5=16, num_5x5=32)
    
    network = inception_module(
            network,pool_filters=64, num_1x1=128, reduce_3x3=128, num_3x3=192, reduce_5x5=32, num_5x5=96)
    
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=3, stride=2)


    network = inception_module(
            network,pool_filters=64, num_1x1=192, reduce_3x3=96, num_3x3=208, reduce_5x5=16, num_5x5=48)

    network = inception_module(
            network,pool_filters=64, num_1x1=160, reduce_3x3=112, num_3x3=224, reduce_5x5=24, num_5x5=64)

    network = inception_module(
            network,pool_filters=64, num_1x1=128, reduce_3x3=128, num_3x3=256, reduce_5x5=24, num_5x5=64)

    network = inception_module(
            network,pool_filters=64, num_1x1=112, reduce_3x3=144, num_3x3=288, reduce_5x5=32, num_5x5=64)

    network = inception_module(
            network,pool_filters=128, num_1x1=256, reduce_3x3=160, num_3x3=320, reduce_5x5=32, num_5x5=128)

    network = lasagne.layers.MaxPool2DLayer(network, pool_size=3, stride=2)

    network = inception_module(
            network,pool_filters=128, num_1x1=256, reduce_3x3=160, num_3x3=320, reduce_5x5=32, num_5x5=128)

    network = inception_module(
            network,pool_filters=128, num_1x1=384, reduce_3x3=192, num_3x3=384, reduce_5x5=48, num_5x5=128)

    network = GlobalPoolLayer(network)

    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.4),
            num_units=344,
            nonlinearity=lasagne.nonlinearities.linear)

    network = lasagne.layers.NonlinearityLayer(network,nonlinearity=lasagne.nonlinearities.softmax)

    return network


input_var = T.tensor4('inputs')
network = build_cnn(input_var)
with np.load('best_model_omit5_v2.npz') as f:
    param_values = [f['arr_%d' % i] for i in range(len(f.files))]
lasagne.layers.set_all_param_values(network, param_values)

prediction = lasagne.layers.get_output(network,deterministic=True)

pred_fn = theano.function([input_var],prediction)

name=pd.read_csv('/home/rqiao/backup/sample_submission.csv',usecols=[0])

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
np.savez('/home/rqiao/backup/modelv2_Resize.npz',test_p)

os.chdir('/scratch/rqiao/Fin/Rotate90')

test_p=[]
for i in name['Image']:
    a=get_img(i)
    test_p.append(pred_fn(a))

print(test_p[:10])
np.savez('/home/rqiao/backup/modelv2_Rotate90.npz',test_p)

os.chdir('/scratch/rqiao/Fin/Rotate180')

test_p=[]
for i in name['Image']:
    a=get_img(i)
    test_p.append(pred_fn(a))

print(test_p[:10])
np.savez('/home/rqiao/backup/modelv2_Rotate180.npz',test_p)


os.chdir('/scratch/rqiao/Fin/Rotate270')

test_p=[]
for i in name['Image']:
    a=get_img(i)
    test_p.append(pred_fn(a))

print(test_p[:10])
np.savez('/home/rqiao/backup/modelv2_Rotate270.npz',test_p)


os.chdir('/scratch/rqiao/Fin/FlipX')

test_p=[]
for i in name['Image']:
    a=get_img(i)
    test_p.append(pred_fn(a))

print(test_p[:10])
np.savez('/home/rqiao/backup/modelv2_FlipX.npz',test_p)






