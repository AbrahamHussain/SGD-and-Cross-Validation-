
import numpy as np 
import csv
import matplotlib.pyplot as plt
import math
import sys
import random

#loads the data from the csv files 
def import_data(dest_file):
    with open(dest_file,'r') as dest_f: 
        data_iter = csv.reader(dest_f, delimiter = ',', quotechar = '"') 
        data = [data for data in data_iter] 
    data_array = np.asarray(data, dtype = str)
    data_array = np.delete(data_array, 0, 0)
    data_array = data_array.astype(np.float)
    temp = np.ones((1000, 1))
    data_array = np.c_[temp, data_array]

    return(data_array)

def squared_loss(x_val, y_val, weights):
    summation = 0
    weights = weights.transpose()
    for i in range(len(x_val)):
        summation += (y_val[i] - np.dot(x_val[i], weights)) ** 2
    return summation 

def weight_update(x_val, y_val, nu, w):
    i = random.randint(0, 999)
    w = w.transpose()
    return (w - nu * (-2 * x_val[i] * (y_val[i] - np.dot(w, x_val[i]))))



def SGD_linear_regression(step, data):
    #arrays that store the info from the csv files 
    w = np.array([0, 0.001, 0.001, 0.001, 0.001]) 
    x_val = np.array([])
    y_val = np.array([])
    train_err = np.array([])
    #loads the info the array
    for i in range(len(data)):
        x_val = np.append(x_val, data[i][0:5])
        y_val = np.append(y_val, data[i][5])
    x_val = np.reshape(x_val, (1000, 5))

    initial_epoch_sqrloss = squared_loss(x_val, y_val, w) 
    delta_result = sys.maxsize 

    for i in range(1000):
        w = weight_update(x_val, y_val, step, w)
    first_epoch_sqrloss = squared_loss(x_val, y_val, w)

    delta_O = first_epoch_sqrloss - initial_epoch_sqrloss
    delta_t = 0 

    t_loss_reduction = initial_epoch_sqrloss
    t_1_loss_reduction = first_epoch_sqrloss
    train_err = np.append(train_err, t_1_loss_reduction)
    
    epoch = 1
    while (abs(delta_result) >= 0.0001):
        epoch += 1
        t_1_loss_reduction = t_loss_reduction
        train_err = np.append(train_err, t_1_loss_reduction)

        for i in range(1000):
            w = weight_update(x_val, y_val, step, w)
        t_loss_reduction = squared_loss(x_val, y_val, w)
        delta_t = t_loss_reduction - t_1_loss_reduction
        delta_result = delta_t / delta_O


    epoch_range = np.arange(epoch)
    return(epoch_range, train_err)

def closed_form(data):
    x_val = np.array([])
    y_val = np.array([])
    for i in range(len(data)):
        x_val = np.append(x_val, data[i][0:5])
        y_val = np.append(y_val, data[i][5])
    x_val = np.reshape(x_val, (1000, 5))
    x_trans = np.transpose(x_val)
    weight = (np.linalg.inv((x_trans.dot(x_val))).dot(x_trans)).dot(y_val)
    print (weight)



         
        
def __main__():
    steps = [(math.e)**-10, (math.e)**-11, (math.e)**-12, (math.e)**-13, (math.e)**-15]
    data = import_data('/Users/AbrahamHussain/Desktop/CS155/Set 1/sgd_data.csv')

    closed_form(data)


    
    
    SGD_linear_regression(steps[-1], data)

    train_10 = []
    epoch_10 = [] 
    train_11 = []
    epoch_11 = [] 
    train_12 = [] 
    epoch_12 = [] 
    train_13 = [] 
    epoch_13 = []  
    train_15 = [] 
    epoch_15 = []

    for i in steps: 
        result = SGD_linear_regression(i, data)
        if(i == (math.e)**-10): 
            train_10 = result[1]
            epoch_10 = result[0]
        if(i == (math.e)**-11):
            train_11 = result[1]
            epoch_11 = result[0]
        if(i == (math.e)**-12):
            train_12 = result[1]
            epoch_12 = result[0]
        if(i == (math.e)**-13):
             train_13 = result[1]
             epoch_13 = result[0]
        if(i == (math.e)**-15):
             train_15 = result[1] 
             epoch_15 = result[0] 

    plt_10 = plt.plot(epoch_10, train_10, color='r', label='e^-10')
    plt_11 = plt.plot(epoch_11, train_11, color='g', label='e^-11')
    plt_12 = plt.plot(epoch_12, train_12, color='b', label='e^-12')
    plt_13 = plt.plot(epoch_13, train_13, color='k', label='e^-13')
    plt_15 = plt.plot(epoch_15, train_15, color='y', label='e^-15')
    plt.title("SGD for linear regression") 
    plt.xlabel("Epoch")
    plt.ylabel("Training Error") 
    plt.legend()
    plt.show()
    



 

__main__()