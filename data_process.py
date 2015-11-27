#!/opt/sharcnet/python/2.7.8/gcc/bin/python

import numpy as np
import pandas as pd
import os
import sys
from PIL import Image
import glob
from collections import Counter

os.chdir('/scratch/rqiao/Fin/Resize')
a=glob.glob('w*.jpg')
os.chdir('..')
train_label=pd.read_csv('train.csv')
ids=train_label['whaleID']
Iname=train_label['Image']
idcount=Counter(ids).most_common()
WNUM=447
Wname=list()
for tup in idcount[:WNUM]:
    Wname.append(tup[0])

Imgname=list()
for i in range(WNUM):
    Imgname.append(list(Iname[ids==Wname[i]]))

target=[]
data=np.zeros((1,3,256,256),dtype=np.float32)
os.chdir('./Resize')
for i in range(WNUM):
    for file in Imgname[i]:
        temp=Image.open(file)
        aray=np.array(temp,dtype=np.float32)
        aray=np.rollaxis(aray,2)
        aray=aray.reshape(1,3,256,256)
        data=np.concatenate((data,aray),0)
        target.append(i)
        
data=data[1:,:,:,:]
data=data/128.0
target=np.array(target,dtype=np.int16)
length=len(target)
arr=np.arange(length)
np.random.shuffle(arr)
train_i=arr[:int(np.floor(length*0.8))]
valid_i=arr[int(np.floor(length*0.8)):int(np.floor(length*0.9))]
test_i=arr[int(np.floor(length*0.9)):]
np.savez("/scratch/rqiao/resize_dataF447.npz",data[train_i,:,:,:],target[train_i],data[valid_i,:,:,:],target[valid_i],data[test_i,:,:,:],target[test_i])

print("successful done")
