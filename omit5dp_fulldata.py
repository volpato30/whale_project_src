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


length=4288
arr=np.arange(length)
np.random.shuffle(arr)
train_i=arr[:int(np.floor(length*0.9))]
valid_i=arr[int(np.floor(length*0.9)):]
train_i=np.concatenate((train_i,train_i+length,train_i+length*2,train_i+length*3,train_i+length*4))
valid_i=np.concatenate((valid_i,valid_i+length,valid_i+length*2,valid_i+length*3,valid_i+length*4))


target=np.zeros(length*5,dtype=np.int16)
data=np.zeros((length*5,3,256,256),dtype=np.float32)
os.chdir('/scratch/rqiao/Fin/Resize')
j=0
for i in range(len(Imgname)):
    for file in Imgname[i]:
        temp=Image.open(file)
        aray=np.array(temp,dtype=np.float32)
        aray=np.rollaxis(aray,2)
        aray=aray.reshape(3,256,256)
        data[j,:,:,:]=aray
        target[j]=i
        j+=1           

os.chdir('/scratch/rqiao/Fin/FlipX')
for i in range(len(Imgname)):
    for file in Imgname[i]:
        temp=Image.open(file)
        aray=np.array(temp,dtype=np.float32)
        aray=np.rollaxis(aray,2)
        aray=aray.reshape(3,256,256)
        data[j,:,:,:]=aray
        target[j]=i
        j+=1           

os.chdir('/scratch/rqiao/Fin/Rotate90')
for i in range(len(Imgname)):
    for file in Imgname[i]:
        temp=Image.open(file)
        aray=np.array(temp,dtype=np.float32)
        aray=np.rollaxis(aray,2)
        aray=aray.reshape(3,256,256)
        data[j,:,:,:]=aray
        target[j]=i
        j+=1           

os.chdir('/scratch/rqiao/Fin/Rotate180')
for i in range(len(Imgname)):
    for file in Imgname[i]:
        temp=Image.open(file)
        aray=np.array(temp,dtype=np.float32)
        aray=np.rollaxis(aray,2)
        aray=aray.reshape(3,256,256)
        data[j,:,:,:]=aray
        target[j]=i
        j+=1           

os.chdir('/scratch/rqiao/Fin/Rotate270')
for i in range(len(Imgname)):
    for file in Imgname[i]:
        temp=Image.open(file)
        aray=np.array(temp,dtype=np.float32)
        aray=np.rollaxis(aray,2)
        aray=aray.reshape(3,256,256)
        data[j,:,:,:]=aray
        target[j]=i
        j+=1           

data=data/128.0


np.savez("/scratch/rqiao/full_dataOmit5.npz",data[train_i,:,:,:],target[train_i],data[valid_i,:,:,:],target[valid_i])

print("successful done")
