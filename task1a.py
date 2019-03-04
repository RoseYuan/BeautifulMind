# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 09:26:16 2019

@author: WH
"""
import numpy as np

import pandas as pd

def h(w,x):
    return np.dot(w,x)

def RMSE(Y, YH):
    return np.sqrt(np.mean(np.square(Y-YH)))

def Cross_validation(data,fold_num):
    row = data.shape[0]
    if row%fold_num == 0:
        num_each_fold = int(row/fold_num)
    if row%fold_num != 0:
        a = row%fold_num
        print('not divisible, eliminate last %d data' % a)
        #data = data[:col-a,:]
        num_each_fold = int(row/fold_num)
    data_cv = []
    data_part = []
    for i in range(fold_num):
        data_part = data[i*num_each_fold:(i+1)*num_each_fold]
        data_part = np.array(data_part)
        data_cv.append(data_part)
    return np.asarray(data_cv, dtype=np.float32)

def rate_update(rate,Rwh,Rw,ca,cd):
    if Rwh<Rw:
        rate = rate -ca
    if Rwh>Rw:
        rate = rate -cd
    
def ridge_gradient(train_data, namuda, accept_error, initial_rate):
    y_data = train_data[:,0]
    x_data = train_data[:,1:]
    column = x_data.shape[1]
    row = x_data.shape[0]
    w_t = np.zeros((1,column))
    w = np.zeros((1,column))
    rate = initial_rate
    
    inter_number = 1
    while inter_number:
        rw = np.zeros((1,column))
        #Rw = np.zeros((1,column))
       # Rwh = np.zeros((1,column))
        for i in range(row):
            y = 0.0
            y = y_data[i]
            x = x_data[i]     
            x = np.array(x,dtype=float)
            #print(y,np.mean(x))
            rw = rw - 2*(y - h(w_t,x))*x 

        #update learning rate
           #Rw = Rw + (y-h(w_t,x))**2
           #Rwh = Rwh + (y-h(w_t-rate*rw,x))**2
           
        pre = y-h(w_t,x)
        w = w_t*(1-2*namuda*rate) - rate*rw/row   #w is w_t+1
        after = y-h(w,x)
        
        if abs(after)- abs(pre) >= 0 :
            print('after %d interation the residual bigger than last interaton.' %inter_number, 'residual is:', pre)
            break
        if abs(after) <= accept_error:
            w_t = w
            print('after %d interation the residual is small enough,' %inter_number ,'residual is:', after)
            break
        else:
            w_t = w
            inter_number +=1
    return w_t

def calculate_yh(w,test_data):
    y_data = test_data[:,0]    
    x_data = test_data[:,1:] 
    row = x_data.shape[0] 
    y_data = y_data.reshape([row,1])
    yh = np.zeros((row,1))
    for i in range(row):
        x = x_data[i]  
        #x = np.array(x,dtype=float)
        yh[i] = h(w,x)
    rmse = RMSE(y_data,yh)
    return rmse
        

data = pd.read_csv('train.csv')
data.set_index(["Id"], inplace=True)
#y_data = train_data.pop('y')
column = data.shape[1]
data_cv = []
data_cv = Cross_validation(data,10) 
namuda = [0.1, 1, 10, 100, 1000]
rmse_namuda = np.zeros((len(namuda),1))
fold_num = 10
for j in range(len(namuda)): #len(namuda)
    n = namuda[j]
    print('nameda is', n )
    fold_rmse = np.zeros((1,fold_num))
    for i in range(fold_num):
        test_data = data_cv[i]
        train_data = np.delete(data_cv,i,axis = 0)
        train_data = np.reshape(train_data,(-1,column))
        w = ridge_gradient(train_data, n, 0.001, 0.00005) 
        fold_rmse[0,i] = calculate_yh(w,test_data)
    #print(fold_rmse)
    rmse_namuda[j,0] = np.mean(fold_rmse)
    print(rmse_namuda)

np.savetxt('sample.csv', rmse_namuda)  

    
