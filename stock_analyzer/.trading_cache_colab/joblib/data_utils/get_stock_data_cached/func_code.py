# first line: 37
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
