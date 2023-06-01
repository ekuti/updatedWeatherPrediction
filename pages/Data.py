import streamlit as st
import pandas as pd
import numpy as np
##import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
pd.set_option('mode.chained_assignment', None)

from keras.models import Sequential
from keras.layers import Dense, SimpleRNN
from keras.optimizers import RMSprop
from keras.callbacks import Callback


humidity = pd.read_csv("humidity.csv")
temp = pd.read_csv("temperature.csv")
pressure = pd.read_csv("pressure.csv")






st.title("Data Used and data sources")

st.markdown(
    """
     Large sets of data got from ______ data studied and used as parameter to train my RNN model includes data for Humidity,
     Temperature, Pressure spaning from 2012-2017 and ranging various cities.
    """
)

st.title("Humidity data")
st.write(humidity)

st.title("Temperature Data")
st.write(temp)

st.title("Pressure Data")
st.write(pressure)


humidity_SF = humidity[['datetime','San Francisco']]
temp_SF = temp[['datetime','San Francisco']]
pressure_SF = pressure[['datetime','San Francisco']]



st.write("humidity san-franciso head shape",humidity_SF.shape)
st.write("temperature san-franciso head shape",temp_SF.shape)
st.write("pressure san-franciso head shape",pressure_SF.shape)

st.write("How many NaN are there in the humidity dataset?",humidity_SF.isna().sum()['San Francisco'])
st.write("How many NaN are there in the temperature dataset?",temp_SF.isna().sum()['San Francisco'])
st.write("How many NaN are there in the pressure dataset?",pressure_SF.isna().sum()['San Francisco'])


st.markdown(
    """
    Given the irregularities in our dataset we are extracting the first ten and the last ten to find out the the space
    and work on data gotten using data for San-francisco as my test model for first case. 
    """
)
hmhead=humidity_SF.head(10)

hmtail=humidity_SF.tail(10)


st.write(hmhead)
st.write(hmtail)