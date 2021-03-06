#!/opt/sharcnet/python/2.7.5/gcc/bin/python

import glob
import scipy.io as sio
import numpy as np
import os
import re

os.chdir('/scratch/amirlk/IMDB/')
dname=glob.glob('Tes*.mat')
a=[]
for i in dname:
    m = re.search('(?<=Testvec)[0-9]*', i)
    a.append(m.group(0))

a=np.array(a,dtype=np.int32)
seq=a.argsort()
fname=[]
for i in seq:
    fname.append(dname[i])

test_data=np.zeros((25000,1,500,300),dtype=np.float32)
j=0
for file in fname:
    temp=sio.loadmat(file)
    for i in range(500):
        a=np.array(temp['tstvec'][i][0],dtype=np.float32)
        a=a.reshape((1,500,300))
        test_data[j]=a
        j+=1

np.savez("/scratch/rqiao/946project/test_data_v2.npz",test_data)