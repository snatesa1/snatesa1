import pandas as pd
import numpy as np
import streamlit as st
import yfinance as yf
from plotly.offline import iplot

@st.cache
def get_sp500_components():
    df = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    df = df[0]
    tickers = df["Symbol"].to_list()
    tick_com_dic = dict(
        zip(df["Symbol"],df["Security"])
    )
    return tickers,tick_com_dic

print (get_sp500_components())
        
                      
