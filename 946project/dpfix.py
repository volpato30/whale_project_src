#!/opt/sharcnet/python/2.7.5/gcc/bin/python


import numpy as np


a=np.load('/scratch/rqiao/946project/train_data.npz')
data=a['arr_0']
target=a['arr_1']

length=len(target)
arr=np.arange(length)
np.random.shuffle(arr)
train_i=arr[:int(np.floor(length*0.8))]
valid_i=arr[int(np.floor(length*0.8)):int(np.floor(length*0.9))]
test_i=arr[int(np.floor(length*0.9)):]
np.savez("/scratch/rqiao/resize_dataF447.npz",data[train_i,:,:,:],target[train_i],data[valid_i,:,:,:],target[valid_i],data[test_i,:,:,:],target[test_i])

print("successful done")
