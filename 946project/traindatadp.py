import glob
import scipy.io as sio
import numpy as np
import os
import re

os.chdir('/scratch/rqiao/Finaldata/')
dname=glob.glob('Tr*.mat')
a=[]
for i in dname:
    m = re.search('(?<=Trainvec)[0-9]*', i)
    a.append(m.group(0))

a=np.array(a,dtype=np.int32)
seq=a.argsort()
fname=[]
for i in seq:
    fname.append(dname[i])


train_data=np.zeros((1,1,500,300),dtype=np.float32)
for file in fname:
    temp=sio.loadmat(file)
    for i in range(500):
        a=np.array(temp['trvec'][i][0],dtype=np.float32)
        a=a.reshape((1,1,500,300))
        train_data=np.concatenate((test_data,a),axis=0)

train_data=train_data[1:,:,:,:]
label=np.array(label['class'],dtype=np.int16)

np.savez("/scratch/rqiao/946project/train_data.npz",test_data)