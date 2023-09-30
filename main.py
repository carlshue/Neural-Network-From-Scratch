import numpy as np
import pandas as pd
#from matplotlib import pyplot as plt


data = pd.read_csv('./train.csv')
#print(data.head())

data = np.array(data)
m, n = data.shape
np.random.shuffle(data)

data_dev = data [0:1000]
Y_dev = data_dev[0]
X_dev = data_dev[1:n]

data_train=data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]

#print(X_train[:, 0].shape)



def init_params():
    w1 = np.random.rand(10, 784) # -0.5
    b1 = np.random.rand(10, 1) # -0.5
    w2 = np.random.rand(10, 10) # -0.5
    b2 = np.random.rand(10, 1) # -0.5
    return w1, b1, w2, b2



def ReLU(Z):
    return np.maximum(0,Z)



def soft_max(Z):
    return np.exp(Z) / np.sum(np.exp(Z))



def one_hot(Y):
    one_hot_y = np.zeros((Y.size, Y.max() +1))
    one_hot_y[np.arange(Y.size), Y] = 1
    one_hot_y = one_hot_y.T
    return one_hot_y



def forward_prop(w1, b1, w2, b2, X):
    z1 = w1.dot(X) + b1
    a1 = ReLU(z1)
    z2 = w2.dot(a1) + b2
    a2 = soft_max(a1)
    return z1, a1, z2, a2



def deriv_ReLU(Z):
    return Z > 0



def backward_prop(z1, a1, z2, a2, w2, X, Y):
    m = Y.size
    one_hot_y = one_hot(Y)
    dz2 = a2- one_hot_y
    dw2 = 1 / m * dz2.dot(a1.T)
    db2 = 1 / m * np.sum(dz2, 2)
    dz1 = w2.T.dot(dz2) * deriv_ReLU(z1)
    dw1 = 1 / m *dz1.dot(X.T)
    db1 = 1 / m * np.sum(dz2, 2)
    return dw1, db1 , dw2, db2



def update_params(w1, b1, w2, b2, dw1, db1, dw2, db2, alpha):
    w1 = w1 - alpha * dw1
    b1 = b1 - alpha * db1
    w2 = w2 - alpha * dw2
    b2 = b2 - alpha * db2
    return w1, b1, w2, b2



def get_predictions(a2):
    return np.argmax(a2,0)



def get_accuracy(preds, Y):
    print(preds, Y)
    return np.sum(preds == Y) / Y.size



def gradient_descent(X, Y, iters, alpha):
    w1, b1, w2, b2 = init_params()
    for i in range(iters):
        z1, a1, z2, a2 = forward_prop(w1, b1, w2, b2, X)
        dw1, db1, dw2, db2 = backward_prop(z1, a1, z2, a2, w2, X, Y)
        w1, b1, w2, b2 = update_params(w1, b1, w2, b2, dw1, db1, dw2, db2, alpha)
        if i % 25 == 0:
            print('iteration: ', i)
            print(' Acc: ', get_accuracy(get_predictions(a2), Y))
    
    return w1, b1, w2, b2


    
w1, b1, w2, b2 = gradient_descent(X_train, Y_train, 100, 0.1)