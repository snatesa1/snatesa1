# main_analyzer.py

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import functions from custom modules
from data_utils import get_stock_data_cached, fetch_fred_macro_indicators_data
from core_logic import (
    calculate_fibonacci_levels,
    run_ma_crossover_strategy,
    run_macd_strategy,
    run_rsi_strategy
)
from news_and_plotting import (
    plot_strategy_performance,
    fetch_news,
    create_macro_stock_visualization
)

# --- Main Application Logic ---
def run_analysis(ticker_symbol, start_date_str, end_date_str, initial_capital_val):
    """
    Main function to run the analysis.
    """
    print(f"Analyzing {ticker_symbol} from {start_date_str} to {end_date_str} with initial capital ${initial_capital_val:,.2f}")

    # Ensure dates are strings for caching and yfinance
    try:
        start_date_str_fmt = pd.to_datetime(start_date_str).strftime('%Y-%m-%d')
        end_date_str_fmt = pd.to_datetime(end_date_str).strftime('%Y-%m-%d')
    except Exception as e:
        print(f"Error formatting dates: {e}. Please use YYYY-MM-DD format.")
        return

    stock_data = get_stock_data_cached(ticker_symbol, start_date_str_fmt, end_date_str_fmt)
    if stock_data is None or stock_data.empty:
        print(f"Could not retrieve data for {ticker_symbol}. Exiting.")
        return

    fib_levels = calculate_fibonacci_levels(stock_data)
    print("\n--- Fibonacci Retracement Levels (Full Period) ---")
    if fib_levels: 
        for name, val in fib_levels.items(): print(f"{name}: {val:.2f}")
    else: print("Could not calculate Fibonacci levels or data was insufficient.")

    # Define strategy parameters
    ma_short_windows = [20, 50] 
    ma_long_windows = [50, 100, 200]
    ma_params = [{'short_window': s, 'long_window': l} for s in ma_short_windows for l in ma_long_windows if s < l]

    macd_params_list = [{'fast_period': 12, 'slow_period': 26, 'signal_period': 9}] # Keep as list
    
    rsi_periods = [14, 21]
    rsi_overbought = [70, 80]
    rsi_oversold = [20, 30]
    rsi_params_list = [{'rsi_period': p, 'overbought_level': ob, 'oversold_level': os} for p in rsi_periods for ob in rsi_overbought for os in rsi_oversold if os < ob]

    strategies_to_evaluate = [
        {"name": "MA Crossover Strategy", "function": run_ma_crossover_strategy, "params_list": ma_params},
        {"name": "MACD Strategy", "function": run_macd_strategy, "params_list": macd_params_list}, # Use macd_params_list
        {"name": "RSI Strategy", "function": run_rsi_strategy, "params_list": rsi_params_list},
    ]

    all_results = []
    best_strategy_overall = None
    # Initialize with negative infinity, or a very small number if comparing non-Sharpe metrics later
    best_sharpe_ratio = -np.inf 

    print("\n--- Backtesting Strategies ---")
    for strategy_config in strategies_to_evaluate:
        print(f"\nEvaluating {strategy_config['name']}...")
        for params in strategy_config['params_list']:
            params_str = ", ".join([f"{k}={v}" for k, v in params.items()])
            print(f"  Parameters: {params_str}")
            
            # Pass initial_capital_val to the strategy function
            metrics, data_with_signals, portfolio_values = strategy_config['function'](stock_data, params, initial_capital_val)
            
            if metrics and portfolio_values is not None:
                metrics['Strategy'] = strategy_config['name']
                metrics['Parameters'] = params_str
                all_results.append(metrics)
                sharpe_val = metrics.get('Sharpe Ratio', np.nan) # Get Sharpe, default to NaN
                print(f"    Total Return: {metrics.get('Total Return (%)', 0):.2f}%, Sharpe Ratio: {sharpe_val if not pd.isna(sharpe_val) else 'N/A'}, Trades: {metrics.get('Number of Trades',0)}")

                if not pd.isna(sharpe_val) and sharpe_val > best_sharpe_ratio: 
                    best_sharpe_ratio = sharpe_val
                    best_strategy_overall = {
                        "name": strategy_config['name'], "params_str": params_str, "metrics": metrics,
                        "data_with_signals": data_with_signals.copy(), "portfolio_values": portfolio_values.copy()
                    }
            else: print(f"    Failed to backtest with parameters: {params_str}")

    if all_results:
        results_df = pd.DataFrame(all_results)
        # Ensure all expected columns are present before selecting
        expected_cols = ['Strategy', 'Parameters', 'Total Return (%)', 'Annualized Return (%)', 'Sharpe Ratio', 'Max Drawdown (%)', 'CAGR (%)', 'Number of Trades']
        cols_to_display = [col for col in expected_cols if col in results_df.columns]
        results_df = results_df[cols_to_display]
        
        if 'Sharpe Ratio' in results_df.columns:
            results_df = results_df.sort_values(by='Sharpe Ratio', ascending=False, na_position='last').reset_index(drop=True)
        
        print("\n\n--- All Strategy Results (Sorted by Sharpe Ratio) ---")
        print(results_df.to_string()) # Use to_string for better console display
    else:
        print("\nNo strategy results to display."); return

    if best_strategy_overall:
        print("\n\n--- Best Performing Strategy (based on Sharpe Ratio) ---")
        print(f"Strategy: {best_strategy_overall['name']}")
        print(f"Parameters: {best_strategy_overall['params_str']}")
        for metric, value in best_strategy_overall['metrics'].items():
            if metric not in ['Strategy', 'Parameters']: 
                print(f"{metric}: {value if not pd.isna(value) else 'N/A'}")
        
        plot_strategy_performance(
            best_strategy_overall['data_with_signals'], best_strategy_overall['portfolio_values'],
            ticker_symbol, best_strategy_overall['name'], best_strategy_overall['params_str'], fib_levels
        )
    else: print("\nCould not determine a best performing strategy (possibly due to all Sharpe Ratios being NaN or too low).")
    
    print("\n--- Fetching Macroeconomic Data and Creating Visualization ---")
    macro_data_dict = fetch_fred_macro_indicators_data() 

    if macro_data_dict: # Check if macro_data_dict is not None and not empty
        # Ensure ticker_symbol is a string for the plotting function
        ticker_symbol_str = ticker_symbol[0] if isinstance(ticker_symbol, list) else str(ticker_symbol)
        create_macro_stock_visualization(macro_data_dict, stock_data, ticker_symbol_str)
    else:
        print("Could not create macro visualization due to data issues or FRED API key problem.")

    fetch_news(ticker_symbol)

    print("\nAnalysis complete. Remember: Past performance is not indicative of future results.")

if __name__ == "__main__":
    # --- FOR GOOGLE COLAB: Set parameters here ---
    # Ensure TICKER is a string for single stock analysis as designed,
    # or the first element of a list if multiple are fetched by data_utils
    TICKER_INPUT = "MSFT"  # Examples: "AAPL", "MSFT", "GOOGL"
                           # If you used a list like ["AAPL", "MSFT"] in data_utils,
                           # the analysis here will still focus on the first one ("AAPL")
                           # due to how get_stock_data_cached flattens data.
                           # For true multi-ticker in one run, the core logic would need significant changes.

    START_DATE = (datetime.now() - timedelta(days=3*365)).strftime('%Y-%m-%d') # 3 years of data
    END_DATE = datetime.now().strftime('%Y-%m-%d') 
    INITIAL_CAPITAL = 100000
    # --- End of parameters ---
    
    # The main analysis function expects a single ticker symbol string for its logic.
    # If TICKER_INPUT is a list, we take the first element.
    # If it's a space-separated string, data_utils handles it, but run_analysis expects one.
    
    effective_run_ticker = TICKER_INPUT
    if isinstance(TICKER_INPUT, list) and TICKER_INPUT:
        effective_run_ticker = TICKER_INPUT[0]
    elif isinstance(TICKER_INPUT, str) and ' ' in TICKER_INPUT:
        effective_run_ticker = TICKER_INPUT.split(' ')[0]
        print(f"Note: Multiple tickers in string '{TICKER_INPUT}'. Analysis will run for the first: '{effective_run_ticker}'.")


    run_analysis(effective_run_ticker, START_DATE, END_DATE, INITIAL_CAPITAL)
