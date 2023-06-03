import streamlit as st

st.set_page_config(
    page_title="Weather Prediction App",
    page_icon="😀",
)

st.title("Weather Prediction Parameter Model")
st.markdown(
    """
    This is a weather parameter prediction model using [Recurrent neural network(RNN)](https://www.ibm.com/topics/recurrent-neural-networks),
    which is a type of artificial neural network which uses sequential data or time series data,trained and tested with large dataset.
    packages used include [pandas](https://pandas.pydata.org) a fast, powerful, flexible and easy to use open source data analysis
    and manipulation tool,built on top of the Python programming language,[matplotlib](https://matplotlib.org) a comprehensive library
    for creating static, animated, and interactive visualizations in Python. Matplotlib makes easy things easy and hard things possible.
    [Numpy](https://numpy.org) for array operations keras from [Tensorflow](https://tensor.org) to train machine learning
    model and [streamlit](https://streamlit.io) for my interactive web app.
    """
)

st.sidebar.success("Select a page above.")

