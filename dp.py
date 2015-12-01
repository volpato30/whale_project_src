#!/opt/sharcnet/python/2.7.8/gcc/bin/python


import glob
import scipy.io as sio
import numpy as np
import os
os.chdir('/scratch/rqiao/mpdata')
dname=glob.glob('Tr*.mat')
train_data=np.zeros((1,1,2000,300),dtype=np.float32)
train_target=[]
for file in dname[:23]:
    temp=sio.loadmat(file)
    for i in range(100):
        train_target.append(temp['y'][i][0])
        a=np.array(temp['trvec'][i][0],dtype=np.float32)
        a=a.reshape((1,1,2000,300))
        train_data=np.concatenate((train_data,a),axis=0)

train_data=train_data[1:,:,:,:]
train_target=np.array(train_target,dtype=np.int16)
train_target=np.array((train_target+1)/2,dtype=np.int16)

valid_data=np.zeros((1,1,2000,300),dtype=np.float32)
valid_target=[]
for file in dname[23:]:
    temp=sio.loadmat(file)
    for i in range(100):
        valid_target.append(temp['y'][i][0])
        a=np.array(temp['trvec'][i][0],dtype=np.float32)
        a=a.reshape((1,1,2000,300))
        valid_data=np.concatenate((valid_data,a),axis=0)

valid_data=valid_data[1:,:,:,:]
valid_target=np.array(valid_target,dtype=np.int16)
valid_target=np.array((valid_target+1)/2,dtype=np.int16)

print(train_data.shape)
print(train_target.shape)
print(valid_data.shape)
print(valid_data.shape)
np.savez("/scratch/rqiao/miniproject.npz",train_data,train_target,valid_data,valid_target)