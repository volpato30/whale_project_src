#!/opt/sharcnet/python/2.7.5/gcc/bin/python

import glob
import scipy.io as sio
import pandas as pd
import numpy as np
import os
import re

os.chdir('/scratch/amirlk/IMDB/')
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

label=pd.read_csv('/scratch/rqiao/Label.txt')
label=np.array(label['class'],dtype=np.int16)


train_data=np.zeros((1,1,500,300),dtype=np.float32)
for file in fname:
    temp=sio.loadmat(file)
    for i in range(500):
        a=np.array(temp['trvec'][i][0],dtype=np.float32)
        a=a.reshape((1,1,500,300))
        train_data=np.concatenate((train_data,a),axis=0)

train_data=train_data[1:,:,:,:]


np.savez("/scratch/rqiao/946project/train_data.npz",train_data,label)