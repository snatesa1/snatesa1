# app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import feedparser
from datetime import datetime, timedelta
import os
from joblib import Memory
from fredapi import Fred
import data_utils
import fundamental_analysis
from core_logic import (
    run_ma_crossover_strategy,
    run_macd_strategy,
    run_rsi_strategy,
    run_custom_supertrend_strategy
)
from news_and_plotting import plot_strategy_performance as plot_strategy_performance_plotly, create_macro_stock_visualization as plot_macro_stock_visualization_plotly, fetch_news

# --- Helper Functions (Adapted from previous modules) ---

def initialize_fred_api():
    """Initialize FRED API with key from Streamlit secrets."""
    try:
        fred_api_key_secret = st.secrets.get("FRED_API_KEY")
        if not fred_api_key_secret:
            st.sidebar.warning("FRED API Key not found in Streamlit secrets. Macroeconomic data from FRED will not be available. "
                               "Please add it to your .streamlit/secrets.toml file.")
            return None
        from fredapi import Fred
        fred = Fred(api_key=fred_api_key_secret)
        return fred
    except Exception as e:
        st.sidebar.error(f"Error initializing FRED API: {e}")
        return None

# --- Data Fetching (from data_utils.py, adapted for Streamlit) ---
def get_stock_data_cached(ticker_symbol_param, start_date_str, end_date_str):
    import yfinance as yf
    import pandas as pd
    effective_ticker = ticker_symbol_param
    if isinstance(ticker_symbol_param, list):
        effective_ticker = ticker_symbol_param[0] if ticker_symbol_param else ""
    if not effective_ticker.strip():
        st.error("Ticker symbol cannot be empty.")
        return None
    try:
        data = yf.download(effective_ticker, start=start_date_str, end=end_date_str, progress=False, auto_adjust=True)
        if data.empty:
            st.warning(f"No data found for '{effective_ticker}' in the given date range.")
            return None
        if isinstance(data.columns, pd.MultiIndex):
            try:
                data = data.xs(effective_ticker, level=1, axis=1)
            except KeyError:
                if effective_ticker in data.columns.get_level_values(0):
                    data = data.xs(effective_ticker, level=0, axis=1)
                else:
                    data.columns = data.columns.droplevel(0)
        if 'Adj Close' not in data.columns:
            if 'Close' in data.columns:
                data['Adj Close'] = data['Close']
            else:
                st.error(f"'Adj Close' and 'Close' columns not found for '{effective_ticker}'. Available: {list(data.columns)}")
                return None
        return data
    except Exception as e:
        st.error(f"Error fetching stock data for '{effective_ticker}': {e}")
        return None

# --- Fibonacci Helper ---
def calculate_fibonacci_levels(data):
    import pandas as pd
    if data is None or data.empty or len(data) < 2 or 'High' not in data.columns or 'Low' not in data.columns:
        return {}
    period_high = data['High'].max()
    period_low = data['Low'].min()
    if pd.isna(period_high) or pd.isna(period_low): return {}
    diff = period_high - period_low
    if diff == 0: return {'0.0% (High)': period_high, '100.0% (Low)': period_low}
    levels = {
        '0.0% (High)': period_high, '23.6%': period_high - 0.236 * diff,
        '38.2%': period_high - 0.382 * diff, '50.0%': period_high - 0.500 * diff,
        '61.8%': period_high - 0.618 * diff, '78.6%': period_high - 0.786 * diff,
        '100.0% (Low)': period_low,
    }
    return levels

# --- Streamlit App UI and Logic ---
def main():
    st.set_page_config(layout="wide", page_title="Stock Strategy Analyzer")
    # State management for active tab
    if 'active_tab' not in st.session_state:
        st.session_state['active_tab'] = 0  # 0 = Strategy Analysis, 1 = Fundamental Analysis

    tab_names = ["Strategy Analysis", "Fundamental Analysis"]
    selected_tab = st.sidebar.radio("Navigation", tab_names, index=st.session_state['active_tab'], key="main_tab_radio")
    tab_idx = tab_names.index(selected_tab)
    st.session_state['active_tab'] = tab_idx

    if tab_idx == 0:
        st.title("📈 Systematic Stock Strategy Analyzer")
        # --- Strategy Summaries Section ---
        import data_utils
        with st.expander('ℹ️ Strategy Summaries'):
            for strat, desc in data_utils.strategy_summaries.items():
                st.markdown(f"**{strat}:** {desc}")

        # --- Sidebar for Inputs ---
        st.sidebar.header("⚙️ Analysis Configuration")
        ticker_symbol = st.sidebar.text_input("Stock Ticker Symbol", value="AAPL").upper()

        # Date inputs
        default_end_date = datetime.now()
        default_start_date = default_end_date - timedelta(days=3*365) # 3 years back
        start_date = st.sidebar.date_input("Start Date", value=default_start_date, max_value=default_end_date - timedelta(days=1))
        end_date = st.sidebar.date_input("End Date", value=default_end_date, min_value=start_date + timedelta(days=1), max_value=default_end_date)

        initial_capital = st.sidebar.number_input("Initial Capital", min_value=1000, value=100000, step=1000)

        # Initialize FRED client (once per session or on demand)
        if 'fred_client' not in st.session_state:
            st.session_state.fred_client = initialize_fred_api()


        # Add a run button with a callback to set the tab
        run_button = st.sidebar.button("🚀 Run Analysis", key="strategy_run_button")
        if run_button:
            st.session_state['active_tab'] = 0


        st.sidebar.markdown("---")
        st.sidebar.markdown("Built with Streamlit & Plotly.")


        # --- Main Analysis Area ---
        if run_button and ticker_symbol:
            start_date_str = start_date.strftime('%Y-%m-%d')
            end_date_str = end_date.strftime('%Y-%m-%d')

            with st.spinner(f"Fetching stock data for {ticker_symbol}..."):
                stock_data = get_stock_data_cached(ticker_symbol, start_date_str, end_date_str)

            if stock_data is None or stock_data.empty:
                st.error(f"Failed to fetch stock data for {ticker_symbol}. Please check the ticker and date range.")
            else:
                st.header(f"Analysis for: {ticker_symbol}")
                st.subheader("Stock Data Overview (Last 5 Days)")
                st.dataframe(stock_data.tail())

                with st.expander("Fibonacci Retracement Levels (Full Period)"):
                    fib_levels = calculate_fibonacci_levels(stock_data)
                    if fib_levels:
                        st.table(pd.DataFrame(list(fib_levels.items()), columns=['Level', 'Price']).style.format({'Price': "{:.2f}"}))
                    else:
                        st.info("Could not calculate Fibonacci levels (insufficient data).")
        
                # --- Strategy Definitions ---
                ma_params_list = [{'short_window': s, 'long_window': l} for s in [20, 50] for l in [50, 100, 200] if s < l]
                macd_params_list = [{'fast_period': 12, 'slow_period': 26, 'signal_period': 9}]
                rsi_params_list = [{'rsi_period': p, 'overbought_level': ob, 'oversold_level': os} 
                                for p in [14, 21] for ob in [70, 80] for os in [20, 30] if os < ob]
                supertrend_params_list = [{'atr_length': 10, 'min_mult': 1.0, 'max_mult': 5.0, 'step': 0.5, 'perf_alpha': 10, 'from_cluster': 'Best'}]

                strategies_to_run = [
                    {"name": "MA Crossover", "function": run_ma_crossover_strategy, "params_list": ma_params_list},
                    {"name": "MACD Strategy", "function": run_macd_strategy, "params_list": macd_params_list},
                    {"name": "RSI Strategy", "function": run_rsi_strategy, "params_list": rsi_params_list},
                    {"name": "Custom SuperTrend", "function": run_custom_supertrend_strategy, "params_list": [
                        {
                            'atr_length': 10,
                            'min_mult': 1.0,
                            'max_mult': 5.0,
                            'step': 0.5,
                            'perf_alpha': 10,
                            'from_cluster': 'Best',
                            'max_iter': 1000
                        }
                    ]},
                ]

                all_strategy_results = []
                best_overall_strategy = None
                best_overall_sharpe = -np.inf

                st.subheader("📈 Strategy Backtesting Results")
                progress_bar = st.progress(0)
                total_strategies_to_test = sum(len(s['params_list']) for s in strategies_to_run)
                tested_count = 0

                for strategy_info in strategies_to_run:
                    st.markdown(f"#### Evaluating: {strategy_info['name']}")
                    for params_config in strategy_info['params_list']:
                        params_config_str = ", ".join([f"{k.replace('_',' ').title()}={v}" for k,v in params_config.items()])
                        with st.spinner(f"Running {strategy_info['name']} with {params_config_str}..."):
                            metrics, data_with_signals, portfolio_values = strategy_info['function'](stock_data, params_config, initial_capital)
                        tested_count += 1
                        progress_bar.progress(tested_count / total_strategies_to_test)

                        if metrics and portfolio_values is not None:
                            metrics['Strategy Name'] = strategy_info['name']
                            metrics['Parameters'] = params_config_str
                            # Entry/Exit/Position columns
                            if data_with_signals is not None and 'position' in data_with_signals.columns:
                                entry_dates = data_with_signals.index[data_with_signals['position'].diff().fillna(0) == 1]
                                exit_dates = data_with_signals.index[data_with_signals['position'].diff().fillna(0) == -1]
                                # Robustly convert to datetime before formatting
                                metrics['Entry Date'] = pd.to_datetime(entry_dates[0]).strftime('%Y-%m-%d') if len(entry_dates) > 0 else 'N/A'
                                metrics['Exit Date'] = pd.to_datetime(exit_dates[-1]).strftime('%Y-%m-%d') if len(exit_dates) > 0 else 'N/A'
                                metrics['Open Position'] = int(data_with_signals['position'].iloc[-1]) if not data_with_signals.empty else 0
                                metrics['Closed Position'] = int((data_with_signals['position'].diff().fillna(0) == -1).sum())
                            else:
                                metrics['Entry Date'] = 'N/A'
                                metrics['Exit Date'] = 'N/A'
                                metrics['Open Position'] = 'N/A'
                                metrics['Closed Position'] = 'N/A'
                            all_strategy_results.append(metrics)
                            current_sharpe = metrics.get('Sharpe Ratio', -np.inf)
                            if pd.notna(current_sharpe) and current_sharpe > best_overall_sharpe:
                                best_overall_sharpe = current_sharpe
                                best_overall_strategy = {
                                    "name": strategy_info['name'],
                                    "params_str": params_config_str,
                                    "metrics": metrics,
                                    "data_with_signals": data_with_signals.copy(),
                                    "portfolio_values": portfolio_values.copy()
                                }
                progress_bar.empty() # Remove progress bar after completion

                if all_strategy_results:
                    results_summary_df = pd.DataFrame(all_strategy_results)
                    # Add new columns to display
                    display_cols = ['Strategy Name', 'Parameters', 'Entry Date', 'Exit Date', 'Open Position', 'Closed Position',
                                    'Total Return (%)', 'Annualized Return (%)', 'Sharpe Ratio', 'Max Drawdown (%)', 'CAGR (%)', 'Number of Trades']
                    results_summary_df = results_summary_df[display_cols]
                    results_summary_df = results_summary_df.sort_values(by='Sharpe Ratio', ascending=False, na_position='last').reset_index(drop=True)
                    
                    st.markdown("##### All Strategy Results Summary (Sorted by Sharpe Ratio)")
                    st.dataframe(results_summary_df.style.format({
                        "Total Return (%)": "{:.2f}%", "Annualized Return (%)": "{:.2f}%",
                        "Sharpe Ratio": "{:.2f}", "Max Drawdown (%)": "{:.2f}%", "CAGR (%)": "{:.2f}%"
                    }).highlight_max(subset=['Sharpe Ratio'], color='lightgreen'))

                    if best_overall_strategy:
                        st.header("🏆 Best Performing Strategy (Based on Sharpe Ratio)")
                        st.subheader(f"{best_overall_strategy['name']} - Parameters: {best_overall_strategy['params_str']}")
                        
                        # Display metrics for best strategy
                        metrics_df = pd.DataFrame([best_overall_strategy['metrics']])[['Total Return (%)', 'Annualized Return (%)', 'Sharpe Ratio', 'Max Drawdown (%)', 'CAGR (%)', 'Number of Trades']]
                        st.table(metrics_df.style.format("{:.2f}"))

                        with st.spinner("Generating strategy performance plot..."):
                            fig_performance, fig_equity = plot_strategy_performance_plotly(
                                best_overall_strategy['data_with_signals'],
                                best_overall_strategy['portfolio_values'],
                                ticker_symbol,
                                best_overall_strategy['name'],
                                best_overall_strategy['params_str'],
                                fib_levels
                            )
                            st.plotly_chart(fig_performance, use_container_width=True)
                            st.plotly_chart(fig_equity, use_container_width=True)
                    else:
                        st.warning("Could not determine a best-performing strategy from the backtests.")
                else:
                    st.info("No strategy backtesting results to display.")

                # --- Macroeconomic Analysis Section ---
                if st.session_state.fred_client:  # Only run if FRED client is available
                    macro_data, macro_summary = data_utils.fetch_fred_macro_indicators_data()
                    if macro_data and any(val is not None for val in macro_data.values()):
                        if macro_summary is not None and not macro_summary.empty:
                            st.subheader("Macroeconomic Indicator Summary (Latest)")
                            st.table(macro_summary)
                        with st.spinner("Generating macroeconomic visualization..."):
                            fig_macro = plot_macro_stock_visualization_plotly(macro_data, stock_data, ticker_symbol)
                            if fig_macro is not None:
                                st.plotly_chart(fig_macro, use_container_width=True)
                            else:
                                st.info("Could not generate macroeconomic visualization (likely insufficient data).")
                    else:
                        st.info("No macroeconomic data fetched or available for visualization.")
                else:
                    st.info("FRED API not initialized. Macroeconomic analysis skipped.")


                # --- News Section ---
                st.header("📰 Latest News Headlines")
                with st.spinner(f"Fetching news for {ticker_symbol}..."):
                    news_articles = fetch_news(ticker_symbol, num_articles=10)
                if news_articles:
                    # Prepare news table
                    news_rows = []
                    for article in news_articles:
                        title = article.get('title', 'No Title')
                        link = article.get('link', '#')
                        pub_time_struct = article.get('published')
                        pub_date_str = (
                            datetime(*pub_time_struct[:6]).strftime('%Y-%m-%d %H:%M')
                            if pub_time_struct else "N/A"
                        )
                        source = article.get('source', 'Unknown')
                        news_rows.append({
                            "Title": f'<a href="{link}" target="_blank">{title}</a>',
                            "Source": source,
                            "Published": pub_date_str
                        })
                    news_df = pd.DataFrame(news_rows)
                    st.markdown(news_df.to_html(escape=False, index=False), unsafe_allow_html=True)
                else:
                    st.info(f"No news articles found for {ticker_symbol}.")
                st.success("Analysis Complete!")

        elif run_button and not ticker_symbol:
            st.error("Please enter a stock ticker symbol.")

        else:
            st.info("Enter analysis parameters in the sidebar and click 'Run Analysis'.")

    elif tab_idx == 1:
        fundamental_analysis.main()

if __name__ == "__main__":
    main()

