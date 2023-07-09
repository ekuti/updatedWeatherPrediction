import streamlit as st
st.set_option('deprecation.showPyplotGlobalUse', False)
import pandas as pd
import numpy as np

#import matplotlib
#matplotlib.use('TkAgg')


import matplotlib.pyplot as plt
#import matplotlib as pl
pd.set_option('mode.chained_assignment', None)

#pl.use('Qt5Agg')

from keras.models import Sequential
from keras.layers import Dense, SimpleRNN
from keras.optimizers import RMSprop
from keras.callbacks import Callback


humidity = pd.read_csv("humidity.csv")
temp = pd.read_csv("temperature.csv")
pressure = pd.read_csv("pressure.csv")

humidity_SF = humidity[['datetime','San Francisco']]
temp_SF = temp[['datetime','San Francisco']]
pressure_SF = pressure[['datetime','San Francisco']]

Tp = 700


train = np.array(humidity_SF['San Francisco'][:Tp])
test = np.array(humidity_SF['San Francisco'][Tp:])


print("Train data length:", train.shape)
print("Test data length:", test.shape)


train=train.reshape(-1,1)
test=test.reshape(-1,1)


##Buildpage starts here
step = 8

# add step elements into train and test
test = np.append(test,np.repeat(test[-1,],step))
train = np.append(train,np.repeat(train[-1,],step))

print("Train data length:", train.shape)
print("Test data length:", test.shape)

def convertToMatrix(data, step):
    X, Y =[], []
    for i in range(len(data)-step):
        d=i+step  
        X.append(data[i:d,])
        Y.append(data[d,])
    return np.array(X), np.array(Y)

trainX,trainY =convertToMatrix(train,step)
testX,testY =convertToMatrix(test,step)

trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

print("Training data shape:", trainX.shape,', ',trainY.shape)
print("Test data shape:", testX.shape,', ',testY.shape)

st.title("Building the Model")
st.markdown(
    "Keras model with SimpleRNN layer We build a simple function to define the RNN model. It uses a single neuron for the output layer because we are predicting a real-valued number here. As activation, it uses the ReLU function. Following arguments are supported.")
st.markdown(
    "Neurons in RNN layer"
)
st.markdown(
    "Embedding length(i.e the length we chose)"
)
st.markdown(
     "Neurons in densly connected layer"
)
st.markdown(
      "learning rate"
)

def build_simple_rnn(num_units=128, embedding=4,num_dense=32,learning_rate=0.001):
    """
    Builds and compiles a simple RNN model
    Arguments:
              num_units: Number of units of a the simple RNN layer
              embedding: Embedding length
              num_dense: Number of neurons in the dense layer followed by the RNN layer
              learning_rate: Learning rate (uses RMSprop optimizer)
    Returns:
              A compiled Keras model.
    """
    model = Sequential()
    model.add(SimpleRNN(units=num_units, input_shape=(1,embedding), activation="relu"))
    model.add(Dense(num_dense, activation="relu"))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer=RMSprop(learning_rate=learning_rate),metrics=['mse'])
    
    return model 

model_humidity = build_simple_rnn(num_units=128,num_dense=32,embedding=8,learning_rate=0.0005)
model_humidity.summary(print_fn=lambda x: st.text(x))

