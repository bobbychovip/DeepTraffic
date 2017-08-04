#-*-coding:UTF-8 -*-
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def get_norm_data():
    df = pd.read_csv('./data/norm_traffic.csv') 
    vehicles = df['vehicles'].as_matrix()
    velocity = df['velocity'].as_matrix()
    vehicles1 = vehicles.reshape((36, 1152))
    velocity1 = velocity.reshape((36, 1152))
    vehicles2 = np.array([np.reshape(x, (288, 4)) for x in vehicles1])
    velocity2 = np.array([np.reshape(x, (288, 4)) for x in velocity1])
    input_data = []
    for i in range(36):
        for j in range(288):
            input_data.append(np.hstack((vehicles2[i][j], velocity2[i][j])))
    return input_data


def create_dataset():
    dataset = get_norm_data()
    look_back = 4
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        dataX.append(dataset[i:(i+look_back)])
        dataY.append(dataset[i+look_back])
    return np.asarray(dataX), np.asarray(dataY)


def split_dataset():
    dataX, dataY = create_dataset()
    X_train = dataX[:10240]
    y_train = dataY[:10240]
    X_test = dataX[10236:]
    y_test = dataY[10236:]
    #X_train, X_test, y_train, y_test = train_test_split(dataX, dataY, test_size=0.1)
    return X_train, y_train, X_test, y_test

#df = pd.read_csv('norm_traffic1.csv')
#print df.shape


#data = get_norm_data()
#print len(data)
#dataX, dataY = create_dataset()
#print dataX.shape, dataY.shape
