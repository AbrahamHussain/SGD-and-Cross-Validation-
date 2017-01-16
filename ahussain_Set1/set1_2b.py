import numpy as np 
from sklearn.model_selection import KFold
import csv
import matplotlib.pyplot as plt

def import_data(dest_file):
    with open(dest_file,'r') as dest_f: 
        data_iter = csv.reader(dest_f, delimiter = ',', quotechar = '"') 
        data = [data for data in data_iter] 
    data_array = np.asarray(data, dtype = str)
    data_array = np.delete(data_array, 0, 0)
    data_array = data_array.astype(np.float)
    return data_array


def poly_regression(arr, d):
    train_err = 0 
    test_err = 0 
    x_val = np.array([])
    y_val = np.array([])
    for i in range(len(arr)):
        x_val = np.append(x_val, arr[i][0])
        y_val = np.append(y_val, arr[i][1])
    kf = KFold(n_splits=5)
    for train_index, test_index in kf.split(arr):
        x_train, x_test = x_val[train_index], x_val[test_index]
        y_train, y_test = y_val[train_index], y_val[test_index]

        poly_coeff = np.polyfit(x_train, y_train, d)

        for i in range(len(y_train)):    
            train_err += ((np.polyval(poly_coeff, x_train[i]) - y_train[i])**2) / float(len(y_train)) 

        for j in range(len(y_test)):
            test_err += ((np.polyval(poly_coeff, x_test[j]) - y_test[j])**2) / float(len(y_test)) 
        
    train_err = train_err / 5.0
    test_err = test_err / 5.0
    
    return(train_err, test_err)
    

def __main__():
    arr = import_data(dest_file = '/Users/AbrahamHussain/Desktop/CS155/Set 1/bv_data.csv')
    N = range(20, 105, 5)
    deg = [1, 2, 6, 12] 

    p1_train = []
    p1_test = []

    p2_train = [] 
    p2_test = []

    p6_train = []
    p6_test = []

    p12_train = []
    p12_test = []  


    for d in deg:
        train_err = []
        test_err = []
        for i in N: 
            temp = poly_regression(arr[:i], d)
            train_err.append(temp[0])
            test_err.append(temp[1])

        if(d == 1): 
            p1_train = train_err
            p1_test = test_err
        if(d == 2): 
            p2_train = train_err
            p2_test = test_err
        if(d == 6): 
            p6_train = train_err
            p6_test = test_err
        if(d == 12): 
            p12_test = test_err
            p12_train = train_err

    


    
    #Change this to plot the other degrees individually 
    plt.plot(N, p1_test, color='r', label='Testing Error')
    plt.plot(N, p1_train, color='b', label='Validation Error')
    plt.title("K-Fold Cross Validation") 
    plt.xlabel("datapoint")
    plt.ylabel("Error") 
    plt.ylim(0,5)
    plt.legend()
    plt.show()


    
__main__()



