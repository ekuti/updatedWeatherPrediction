import streamlit as st
from PIL import Image

image = Image.open('8.png')

st.image(image, caption='Sunrise by the mountains')