import streamlit as st
import yfinance as yf
import pandas as pd
from llm_utils import LLM_MODELS, initialize_llm_client, get_llm_response
import time
from functools import lru_cache

# --- Simple in-memory rate limiter ---
class RateLimiter:
    def __init__(self, min_interval_sec):
        self.min_interval_sec = min_interval_sec
        self.last_call = 0
    def wait(self):
        now = time.time()
        elapsed = now - self.last_call
        if elapsed < self.min_interval_sec:
            time.sleep(self.min_interval_sec - elapsed)
        self.last_call = time.time()

stock_api_limiter = RateLimiter(1.1)  # yfinance: 1 call/sec
llm_api_limiter = RateLimiter(3.1)    # LLM: 1 call/3 sec (adjust as needed)

@lru_cache(maxsize=32)
def get_yf_info(ticker):
    stock_api_limiter.wait()
    stock = yf.Ticker(ticker)
    return stock.info, stock.financials

@lru_cache(maxsize=64)
def cached_llm_response(prompt, context, provider, model_name, api_key):
    llm_api_limiter.wait()
    # Re-init client each time for cache safety
    client = initialize_llm_client(provider, api_key)
    messages = [{"role": "user", "content": prompt}]
    return get_llm_response(client, provider, model_name, messages, context)

def initialize_chat_state():
    if 'fa_chat_messages' not in st.session_state:
        st.session_state['fa_chat_messages'] = []
    if 'fa_llm_client' not in st.session_state:
        st.session_state['fa_llm_client'] = None
    if 'fa_llm_model' not in st.session_state:
        st.session_state['fa_llm_model'] = st.secrets.get('LLM_MODEL', 'GPT-4')
    if 'fa_api_key' not in st.session_state:
        st.session_state['fa_api_key'] = st.secrets.get('OPENAI_API_KEY', '')
    if 'fa_run_analysis' not in st.session_state:
        st.session_state['fa_run_analysis'] = False

def main():
    st.title("📊 Fundamental Analysis")
    initialize_chat_state()

    # --- Sidebar Configuration ---
    st.sidebar.header("Fundamental Analysis Configuration")
    ticker_symbol = st.sidebar.text_input("Stock Ticker Symbol", value="AAPL", key="fa_ticker").upper()
    run_button = st.sidebar.button("🔍 Run Fundamental Analysis", key="fa_run_button")
    if run_button:
        st.session_state['fa_run_analysis'] = True

    # --- API Key: hidden from sidebar, use from secrets or session state only ---
    api_key = st.secrets.get('OPENAI_API_KEY', '')
    if api_key:
        st.session_state['fa_api_key'] = api_key
    selected_model = st.secrets.get('LLM_MODEL', 'GPT-4')
    st.session_state['fa_llm_model'] = selected_model
    provider = LLM_MODELS[selected_model]["provider"]
    model_name = LLM_MODELS[selected_model]["model"]
    # LLM client state
    if api_key and (not st.session_state['fa_llm_client'] or st.session_state.get('fa_llm_client_provider') != provider or st.session_state.get('fa_llm_client_api_key') != api_key):
        st.session_state['fa_llm_client'] = initialize_llm_client(provider, api_key)
        st.session_state['fa_llm_client_provider'] = provider
        st.session_state['fa_llm_client_api_key'] = api_key

    # --- Main Panel: Fundamental Analysis ---
    chat_display = []
    if st.session_state['fa_run_analysis'] and ticker_symbol:
        st.header(f"Fundamental Analysis for: {ticker_symbol}")
        try:
            info, financials = get_yf_info(ticker_symbol)
            # Display key metrics
            metrics = {
                "Market Cap": info.get('marketCap', 'N/A'),
                "P/E Ratio": info.get('trailingPE', 'N/A'),
                "EPS": info.get('trailingEps', 'N/A'),
                "52 Week High": info.get('fiftyTwoWeekHigh', 'N/A'),
                "52 Week Low": info.get('fiftyTwoWeekLow', 'N/A'),
                "Dividend Yield": info.get('dividendYield', 'N/A'),
            }
            st.subheader("Key Metrics")
            metrics_df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])
            st.table(metrics_df)
            # Financial Statements
            st.subheader("Financial Statements")
            if not financials.empty:
                st.dataframe(financials)
            else:
                st.info("No financial statements available")
        except Exception as e:
            st.error(f"Error fetching data for {ticker_symbol}: {e}")
        chat_display = True
    elif st.session_state['fa_run_analysis'] and not ticker_symbol:
        st.error("Please enter a stock ticker symbol.")
        chat_display = True
    else:
        chat_display = False

    # --- Main Panel: Chat Interface at Bottom ---
    if chat_display:
        st.markdown("---")
        st.subheader("💬 AI Analysis Assistant")
        # Display chat messages
        for message in st.session_state['fa_chat_messages']:
            role = message["role"]
            if role == "user":
                st.markdown(f"**You:** {message['content']}")
            else:
                st.markdown(f"**AI:** {message['content']}")
        # Chat input
        chat_prompt = st.text_input("Type your question and press Enter", key="fa_chat_input")
        chat_button = st.button("Send", key="fa_chat_send")
        if chat_button and chat_prompt:
            if not st.session_state['fa_llm_client']:
                st.warning("No valid API key found in secrets.toml.")
            else:
                st.session_state['fa_chat_messages'].append({"role": "user", "content": chat_prompt})
                with st.spinner("AI is thinking..."):
                    context = f"You are a financial analysis expert. The current stock being analyzed is {ticker_symbol}."
                    # Use cache for LLM response
                    response = cached_llm_response(chat_prompt, context, provider, model_name, api_key)
                    st.session_state['fa_chat_messages'].append({"role": "assistant", "content": response})
                st.rerun()
        if st.button("🗑️ Clear Chat History", key="fa_clear_chat"):
            st.session_state['fa_chat_messages'] = []
            st.rerun()
