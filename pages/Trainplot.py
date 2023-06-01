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


humidity = pd.read_csv("humidity.csv")
temp = pd.read_csv("temperature.csv")
pressure = pd.read_csv("pressure.csv")

humidity_SF = humidity[['datetime','San Francisco']]
temp_SF = temp[['datetime','San Francisco']]
pressure_SF = pressure[['datetime','San Francisco']]

Tp = 700


st.title("TrainPlots for Temperature,Humidity and Pressure")

st.markdown(
    """
     This are the train plot gotten from the program when the data sets were pass through the funtion
     With my Tp = 700
    """
)
def plot_train_points(quantity='humidity',Tp=7000):
    plt.figure(figsize=(10,5))
    if quantity=='humidity':
        plt.title("Humidity of first {} data points".format(Tp),fontsize=30)
        plt.plot(humidity_SF['San Francisco'][:Tp],c='k',lw=1)
    if quantity=='temperature':
        plt.title("Temperature of first {} data points".format(Tp),fontsize=30)
        plt.plot(temp_SF['San Francisco'][:Tp],c='k',lw=1)
    if quantity=='pressure':
        plt.title("Pressure of first {} data points".format(Tp),fontsize=30)
        plt.plot(pressure_SF['San Francisco'][:Tp],c='k',lw=1)
    plt.grid(True)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.show()

plot_train_points('humidity')
st.pyplot()
    

plot_train_points('temperature')
st.pyplot()

plot_train_points('pressure')

st.pyplot()