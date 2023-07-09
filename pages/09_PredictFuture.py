import streamlit as st
st.set_option('deprecation.showPyplotGlobalUse', False)
import pandas as pd
import numpy as np

#import matplotlib
#matplotlib.use('TkAgg')

import matplotlib.pyplot as plt

pd.set_option('mode.chained_assignment', None)



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


#st.text("Train data length:", train.shape)
#st.text("Test data length:", test.shape)


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

#print("Training data shape:", trainX.shape,', ',trainY.shape)
#print("Test data shape:", testX.shape,', ',testY.shape)

st.title("Now predict the future points")
st.markdown(
    "Now, we can generate predictions for the future by passing testX to the trained model."
    " "
    "#See the magic!"
    "When we plot the predicted vector, we see it matches closely the true values and that is amazing given how little training data was used and how far in the future it had to predict. Time-series techniques like ARIMA, Exponential smoothing, cannot predict very far into the future and their confidence interval quickly grows beyond being useful."
    "Note carefully how the model is able to predict sudden increase in humidity around time-points 12000. There was no indication of such shape or pattern of the data in the training set, yet, it is able to predict the general shape pretty well from the first 7000 data points"
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
#model_humidity.summary(print_fn=lambda x: st.text(x))

class MyCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if (epoch+1) % 50 == 0 and epoch>0:
            ""
            #print("Epoch number {} done".format(epoch+1))

batch_size=8
num_epochs = 1000


#model_humidity.fit(trainX,trainY, 
          #epochs=num_epochs, 
          #batch_size=batch_size, 
          #callbacks=[MyCallback()],verbose=0)


#model_humidity.fit(trainX,trainY, 
          #epochs=num_epochs, 
          #batch_size=batch_size, 
          #callbacks=[MyCallback()],verbose=0)
#plt.figure(figsize=(7,5))
#plt.title("RMSE loss over epochs",fontsize=16)
#plt.plot(np.sqrt(model_humidity.history.history['loss']),c='k',lw=2)
#plt.grid(True)
#plt.xlabel("Epochs",fontsize=14)
#plt.ylabel("Root-mean-squared error",fontsize=14)
#plt.xticks(fontsize=14)
#plt.yticks(fontsize=14)
#plt.show()          




trainPredict = model_humidity.predict(trainX)
testPredict= model_humidity.predict(testX)
predicted=np.concatenate((trainPredict,testPredict),axis=0)





plt.figure(figsize=(10,4))
plt.title("This is what the model predicted",fontsize=18)
plt.plot(testPredict,c='orange')
plt.grid(True)
plt.show()
