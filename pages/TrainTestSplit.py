import streamlit as st
st.set_option('deprecation.showPyplotGlobalUse', False)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
pd.set_option('mode.chained_assignment', None)

from keras.models import Sequential
from keras.layers import Dense, SimpleRNN
from keras.optimizers import RMSprop
from keras.callbacks import Callback

Tp=700

humidity = pd.read_csv("humidity.csv")
temp = pd.read_csv("temperature.csv")
pressure = pd.read_csv("pressure.csv")

humidity_SF = humidity[['datetime','San Francisco']]
temp_SF = temp[['datetime','San Francisco']]
pressure_SF = pressure[['datetime','San Francisco']]
##started here
humidity_SF.interpolate(inplace=True)
humidity_SF.dropna(inplace=True)

temp_SF.interpolate(inplace=True)
temp_SF.dropna(inplace=True)

pressure_SF.interpolate(inplace=True)
pressure_SF.dropna(inplace=True)

print(humidity_SF.shape)
print(temp_SF.shape)
print(pressure_SF.shape)


train = np.array(humidity_SF['San Francisco'][:Tp])
test = np.array(humidity_SF['San Francisco'][Tp:])


print("Train data length:", train.shape)
print("Test data length:", test.shape)


train=train.reshape(-1,1)
test=test.reshape(-1,1)

st.markdown("This is the test train plot acquired from ploting test againts plot make it interactive and show source code and write about it")

plt.figure(figsize=(10,3))
plt.title("Train and test data plotted together",fontsize=16)
plt.plot(np.arange(Tp),train,c='blue')
plt.plot(np.arange(Tp,45252),test,c='orange',alpha=0.7)
plt.legend(['Train','Test'])
plt.grid(True)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()
st.pyplot()
