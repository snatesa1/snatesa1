# data_utils.py

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os
from joblib import Memory
from fredapi import Fred
import streamlit as st

# --- Configuration for Data Utils ---
CACHE_DIR = "./.trading_cache_colab"
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)
memory = Memory(CACHE_DIR, verbose=0)

def initialize_fred_api():
    """Initialize FRED API with key from Streamlit secrets."""
    try:
        fred_api_key = st.secrets.get("FRED_API_KEY")
        if not fred_api_key:
            print("FRED_API_KEY not found in Streamlit secrets.")
            print("Please ensure you have added your FRED API key to .streamlit/secrets.toml with name 'FRED_API_KEY'")
            return None
        fred = Fred(api_key=fred_api_key)
        print("FRED API initialized successfully.")
        return fred
    except Exception as e:
        print(f"Error initializing FRED API: {e}")
        print("Please ensure you have added your FRED API key to .streamlit/secrets.toml with name 'FRED_API_KEY'")
        return None

@memory.cache
def get_stock_data_cached(ticker_symbol_param, start_date, end_date):
    """
    Fetches stock data using yfinance and caches the result.
    Dates should be in 'YYYY-MM-DD' string format.
    Ensures yfinance is called with a string ticker if a list of one ticker is provided,
    and attempts to handle cases where MultiIndex columns might be returned.
    """
    effective_ticker = ticker_symbol_param
    is_single_ticker_intent = True

    if isinstance(ticker_symbol_param, list):
        if len(ticker_symbol_param) == 1:
            effective_ticker = ticker_symbol_param[0]
            print(f"Note: ticker_symbol was a list {ticker_symbol_param}, using string '{effective_ticker}' for yf.download to ensure flat columns.")
        else:
            effective_ticker = ticker_symbol_param[0] # Use first ticker
            is_single_ticker_intent = False
            print(f"Warning: Multiple tickers provided in list {ticker_symbol_param}. Using first ticker '{effective_ticker}'. The script is designed for single-ticker analysis.")
    elif isinstance(ticker_symbol_param, str) and len(ticker_symbol_param.split()) > 1:
        is_single_ticker_intent = False # Multiple tickers are implied
        print(f"Note: ticker_symbol '{ticker_symbol_param}' contains multiple space-separated tickers. yf.download will produce MultiIndex columns.")

    print(f"Fetching data for '{effective_ticker}' from {start_date} to {end_date}...")
    try:
        data = yf.download(effective_ticker, start=start_date, end=end_date, progress=False)
        
        if data.empty:
            print(f"No data found for '{effective_ticker}' in the given date range.")
            return None
        
        if isinstance(data.columns, pd.MultiIndex):
            print(f"Data for '{effective_ticker}' has MultiIndex columns: {data.columns.names}. Attempting to flatten for single ticker analysis.")
            try:
                main_ticker_to_select = effective_ticker.split()[0] # Use the first part of the string (even if it was already single)
                
                if main_ticker_to_select in data.columns.get_level_values(1):
                     data = data.xs(main_ticker_to_select, level=1, axis=1)
                     print(f"Successfully flattened MultiIndex columns to use data for ticker '{main_ticker_to_select}'.")
                else: 
                    print(f"Ticker '{main_ticker_to_select}' not found in MultiIndex level 1. Attempting to drop level 0 of MultiIndex if it's the ticker level.")
                    # Check if the first level contains the ticker symbols
                    if main_ticker_to_select in data.columns.get_level_values(0):
                        data = data.xs(main_ticker_to_select, level=0, axis=1)
                        print(f"Successfully flattened MultiIndex columns by selecting from level 0 for ticker '{main_ticker_to_select}'.")
                    else:
                        print(f"Ticker '{main_ticker_to_select}' not found in MultiIndex level 0 either. Dropping the outermost level (level 0) of MultiIndex as a fallback.")
                        data.columns = data.columns.droplevel(0) # Common for ('Ticker', 'Value') structure if level 1 was not it.
            except Exception as e_flatten:
                print(f"Could not automatically flatten MultiIndex columns for '{effective_ticker}': {e_flatten}. Proceeding with MultiIndex, may cause errors in downstream functions.")

        if 'Adj Close' not in data.columns:
            if 'Close' in data.columns:
                data['Adj Close'] = data['Close']
            else:
                print(f"Error: 'Adj Close' and 'Close' columns not found for '{effective_ticker}' after processing. Available columns: {data.columns}")
                return None
        return data
    except Exception as e:
        print(f"Error in get_stock_data_cached for '{effective_ticker}': {e}")
        return None

def fetch_fred_macro_indicators_data():
    """Fetches key macroeconomic indicators from FRED and returns a dict and a summary table for Streamlit display."""
    fred = initialize_fred_api()
    if fred is None:
        return None, None

    fred_series = {
        "Unemployment Rate": "UNRATE",
        "CPI (YoY Change)": "CPIAUCSL",
        "2-Year Treasury": "GS2"
    }

    macro_data_dict = {}
    summary_rows = []
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*5)

    for name, series_id in fred_series.items():
        try:
            data = fred.get_series(series_id, observation_start=start_date.strftime('%Y-%m-%d'), observation_end=end_date.strftime('%Y-%m-%d'))
            if data is None or data.empty:
                macro_data_dict[name] = None
                summary_rows.append({"Indicator": name, "Latest Value": "N/A", "Change": "N/A"})
                continue
            data = data.dropna()
            if data.empty:
                macro_data_dict[name] = None
                summary_rows.append({"Indicator": name, "Latest Value": "N/A", "Change": "N/A"})
                continue
            if "CPI" in name:
                data_yoy = data.pct_change(periods=12) * 100
                data_yoy = data_yoy.dropna()
                if not data_yoy.empty:
                    macro_data_dict[name] = data_yoy
                    latest_value = data_yoy.iloc[-1]
                    prev_value = data_yoy.iloc[-2] if len(data_yoy) >= 2 else None
                    change_str = f"{latest_value - prev_value:.2f}pp" if prev_value is not None else "N/A"
                    summary_rows.append({"Indicator": name, "Latest Value": f"{latest_value:.2f}%", "Change": change_str})
                else:
                    macro_data_dict[name] = None
                    summary_rows.append({"Indicator": name, "Latest Value": "N/A", "Change": "N/A"})
            else:
                macro_data_dict[name] = data
                latest_value = data.iloc[-1]
                prev_value = data.iloc[-2] if len(data) >= 2 else None
                if prev_value is not None:
                    change = latest_value - prev_value
                    change_pct = (change / prev_value) * 100 if prev_value != 0 else 0
                    change_str = f"{change:.2f}pp, {change_pct:.2f}%"
                else:
                    change_str = "N/A"
                summary_rows.append({"Indicator": name, "Latest Value": f"{latest_value:.2f}%", "Change": change_str})
        except Exception as e:
            macro_data_dict[name] = None
            summary_rows.append({"Indicator": name, "Latest Value": "N/A", "Change": "N/A"})
    summary_table = pd.DataFrame(summary_rows)
    return macro_data_dict, summary_table

# --- Strategy Summaries ---
# Moving Average (MA):
# A moving average smooths out price data by creating a constantly updated average price. It helps identify the direction of the trend.

# MACD (Moving Average Convergence Divergence):
# MACD is a trend-following momentum indicator that shows the relationship between two moving averages of a security’s price. It helps spot changes in the strength, direction, momentum, and duration of a trend.

# RSI (Relative Strength Index):
# RSI is a momentum oscillator that measures the speed and change of price movements. It ranges from 0 to 100 and is used to identify overbought or oversold conditions.

# Fibonacci Retracement:
# Fibonacci retracement levels are horizontal lines that indicate where support and resistance are likely to occur, based on the key Fibonacci numbers identified by mathematician Leonardo Fibonacci.

# --- Strategy Summaries for UI ---
strategy_summaries = {
    "Moving Average (MA)": "A moving average smooths out price data by creating a constantly updated average price. It helps identify the direction of the trend.",
    "MACD (Moving Average Convergence Divergence)": "MACD is a trend-following momentum indicator that shows the relationship between two moving averages of a security’s price. It helps spot changes in the strength, direction, momentum, and duration of a trend.",
    "RSI (Relative Strength Index)": "RSI is a momentum oscillator that measures the speed and change of price movements. It ranges from 0 to 100 and is used to identify overbought or oversold conditions.",
    "Fibonacci Retracement": "Fibonacci retracement levels are horizontal lines that indicate where support and resistance are likely to occur, based on the key Fibonacci numbers identified by mathematician Leonardo Fibonacci."
}
