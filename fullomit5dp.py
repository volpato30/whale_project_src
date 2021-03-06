#!/opt/sharcnet/python/2.7.8/gcc/bin/python


import numpy as np
import pandas as pd
import os
import sys
from PIL import Image
import glob
from collections import Counter

os.chdir('/scratch/rqiao/Fin/Resize')
os.chdir('..')
train_label=pd.read_csv('train.csv')
ids=train_label['whaleID']
Iname=train_label['Image']
idcount=Counter(ids).most_common()
def name_to_num(iname):
    a=np.array(train_label['Image']==iname,dtype=np.int8)
    wid=train_label['whaleID'][a.nonzero()[0][0]]
    for tup in idcount:
        if tup[0]==wid:
            return tup[1]

def id_to_num(wid):
    for tup in idcount:
        if tup[0]==wid:
            return tup[1]

w_seq=[]
for tup in idcount:
    w_seq.append(tup[0])
w_seq.sort()
Imgname=list()
omit=[]
for i in range(len(w_seq)):
    if id_to_num(w_seq[i])<5:
        omit.append(i)
        continue
    Imgname.append(list(Iname[ids==w_seq[i]]))

target=[]
data=np.zeros((1,3,256,256),dtype=np.float32)
os.chdir('/scratch/rqiao/Fin/Resize')
for i in range(len(Imgname)):
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
np.savez("/scratch/rqiao/full_data/resize_dataOmit5.npz",data,target)

del data
del target

target=[]
data=np.zeros((1,3,256,256),dtype=np.float32)
os.chdir('/scratch/rqiao/Fin/FlipX')
for i in range(len(Imgname)):
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
np.savez("/scratch/rqiao/full_data/FlipX_dataOmit5.npz",data,target)


del data
del target

target=[]
data=np.zeros((1,3,256,256),dtype=np.float32)
os.chdir('/scratch/rqiao/Fin/Rotate90')
for i in range(len(Imgname)):
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
np.savez("/scratch/rqiao/full_data/Rotate90_dataOmit5.npz",data,target)

del data
del target

target=[]
data=np.zeros((1,3,256,256),dtype=np.float32)
os.chdir('/scratch/rqiao/Fin/Rotate180')
for i in range(len(Imgname)):
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
np.savez("/scratch/rqiao/full_data/Rotate180.npz",data,target)

del data
del target

target=[]
data=np.zeros((1,3,256,256),dtype=np.float32)
os.chdir('/scratch/rqiao/Fin/Rotate270')
for i in range(len(Imgname)):
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
np.savez("/scratch/rqiao/full_data/Rotate270.npz",data,target)


print("successful done")
<<<<<<< HEAD

=======
>>>>>>> ae5730b3662c4b5d1d9fc547cd720e046c5aee05
