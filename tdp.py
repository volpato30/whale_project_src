#!/opt/sharcnet/python/2.7.8/gcc/bin/python


import glob
import scipy.io as sio
import numpy as np
import os
os.chdir('/scratch/rqiao/mptest')

dname=glob.glob('T*.mat')
dname.sort()
fname=[]
seq=[8,1,9,10,11,12,13,14,15,16,0,2,3,4,5,6,7]
for i in seq:
    fname.append(dname[i])

print(fname)

test_data=np.zeros((1,1,2000,300),dtype=np.float32)
for file in fname:
    temp=sio.loadmat(file)
    for i in range(100):
        a=np.array(temp['tstvec'][i][0],dtype=np.float32)
        a=a.reshape((1,1,2000,300))
        test_data=np.concatenate((test_data,a),axis=0)

test_data=test_data[1:,:,:,:]

np.savez("/scratch/rqiao/mptest_data.npz",test_data)