#!/opt/sharcnet/python/2.7.8/gcc/bin/python
import numpy as np


a=np.load("/scratch/rqiao/full_data/FlipX_dataOmit5.npz")
target=a['arr_1']
print target.shape