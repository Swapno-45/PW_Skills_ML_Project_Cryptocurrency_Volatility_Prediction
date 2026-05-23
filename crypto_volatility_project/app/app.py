
import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.title("Cryptocurrency Volatility Prediction")

st.write("Enter market details to predict volatility")

open_price = st.number_input("Open Price")
high_price = st.number_input("High Price")
low_price = st.number_input("Low Price")
close_price = st.number_input("Close Price")
volume = st.number_input("Volume")
market_cap = st.number_input("Market Cap")

if st.button("Predict"):
    volatility = (high_price - low_price) / open_price

    st.success(f"Predicted Volatility: {round(volatility, 4)}")
