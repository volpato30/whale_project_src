#!/opt/sharcnet/python/2.7.5/gcc/bin/python
import sys
import os
import time
import numpy as np
import scipy.io as sio

a=np.load("/scratch/rqiao/IMDB.npz")
train_target=a['arr_1']
valid_target=a['arr_3']
test_target=a['arr_5']
test_p=np.zeros(25000,dtype=np.int16)
for i in range(20000):
    test_p[i]=train_target[i]
for i in range(2500):
    test_p[i+20000,:]=valid_target[i]
for i in range(2500):
    test_p[i+22500,:]=test_target[i]

sio.savemat('label.mat', {'label': test_p})
