# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset_train = pd.read_csv('input.csv')
x_train, y_train=dataset_train.iloc[:,0:3].values, dataset_train.iloc[:,3:].values
x_train,y_train=np.array(x_train), np.array(y_train)
x_train= np.reshape(x_train, (x_train.shape[0], x_train.shape[1],1))

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import SimpleRNN

model = Sequential()
model.add(SimpleRNN(units = 150,return_sequences=True, input_shape = (3,1)))
model.add(SimpleRNN(units = 50))
model.add(Dense(units = 1)) 

model.compile(optimizer='adam', loss='mean_squared_error')

model.summary()

model.fit(x_train, epochs=2)
predicted_number = model.predict(x_train)