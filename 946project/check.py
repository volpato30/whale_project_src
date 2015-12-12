#!/opt/sharcnet/python/2.7.5/gcc/bin/python


import numpy as np

def load_data():
    
    a=np.load("/scratch/rqiao/IMDB.npz")
    train_data=a['arr_0']
    train_target=a['arr_1']
    valid_data=a['arr_2']
    valid_target=a['arr_3']
    test_data=a['arr_4']
    test_target=a['arr_5']
    return train_data,train_target,valid_data,valid_target,test_data,test_target


datasets = load_data()
X_train, y_train = datasets[0], datasets[1]
X_val, y_val = datasets[2], datasets[3]
X_test, y_test = datasets[4], datasets[5]

print X_train.shape
print y_train.shape
print X_val.shape
print y_val.shape
print X_test.shape
print y_test.shape