# core_logic.py

import pandas as pd
import numpy as np

# Risk-free rate for Sharpe Ratio (annualized)
RISK_FREE_RATE = 0.02

# --- Indicator Calculations ---
def calculate_sma(data, window):
    """Calculates Simple Moving Average."""
    return data['Adj Close'].rolling(window=window).mean()

def calculate_ema(data, window):
    """Calculates Exponential Moving Average."""
    return data['Adj Close'].ewm(span=window, adjust=False).mean()

def calculate_macd(data, fast_period=12, slow_period=26, signal_period=9):
    """Calculates MACD, MACD Signal, and MACD Histogram."""
    ema_fast = calculate_ema(data, window=fast_period)
    ema_slow = calculate_ema(data, window=slow_period)
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def calculate_rsi(data, window=14):
    """Calculates Relative Strength Index (RSI)."""
    delta = data['Adj Close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    
    rs = gain / loss.replace(0, 0.000001) 
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_fibonacci_levels(data):
    """Calculates Fibonacci Retracement levels based on the period's high and low."""
    if not isinstance(data, pd.DataFrame) or data.empty or len(data) < 2:
        if not isinstance(data, pd.DataFrame): print("Fibonacci: Input is not a DataFrame.")
        elif data.empty: print("Fibonacci: Input data is empty.")
        else: print("Fibonacci: Input data has less than 2 rows.")
        return {}

    if 'High' not in data.columns or 'Low' not in data.columns:
        print(f"Fibonacci: 'High' or 'Low' column not in data. Columns: {data.columns}")
        return {}
    
    if not isinstance(data['High'], pd.Series) or not isinstance(data['Low'], pd.Series):
        print(f"Fibonacci: data['High'] (type {type(data['High'])}) or data['Low'] (type {type(data['Low'])}) is not a Series.")
        return {}

    period_high = data['High'].max()
    period_low = data['Low'].min()

    if pd.isna(period_high) or pd.isna(period_low):
        print("Fibonacci: period_high or period_low is NaN. Cannot calculate levels.")
        return {}

    diff = period_high - period_low

    if diff == 0: 
        return {
            '0.0% (High)': period_high,
            '50.0%': period_high, 
            '100.0% (Low)': period_low,
        }

    levels = {
        '0.0% (High)': period_high,
        '23.6%': period_high - 0.236 * diff,
        '38.2%': period_high - 0.382 * diff,
        '50.0%': period_high - 0.500 * diff,
        '61.8%': period_high - 0.618 * diff,
        '78.6%': period_high - 0.786 * diff, 
        '100.0% (Low)': period_low,
    }
    return levels

# --- Strategy Signal Generation & Backtesting ---
def backtest_strategy(data, initial_capital=100000):
    """
    Core backtesting logic.
    """
    if 'signal' not in data.columns:
        print("Error: 'signal' column not found in data for backtesting.")
        return None, None

    capital = initial_capital
    shares = 0
    portfolio_values = pd.Series(index=data.index, dtype=float)
    
    data['position'] = 0 
    current_pos_state = 0 
    
    for i in range(len(data)):
        if data['signal'].iloc[i] == 1:
            current_pos_state = 1
        elif data['signal'].iloc[i] == -1:
            current_pos_state = 0
        data.loc[data.index[i], 'position'] = current_pos_state
        
    data['position'] = data['position'].shift(1).fillna(0)

    for i in range(len(data)):
        adj_close_price = data['Adj Close'].iloc[i]
        
        if data['position'].iloc[i] == 1 and shares == 0: # Buy
            if capital > 0 and adj_close_price > 0: 
                shares_to_buy = capital / adj_close_price
                shares = shares_to_buy
                capital = 0 
        elif data['position'].iloc[i] == 0 and shares > 0: # Sell
            if adj_close_price > 0: 
                capital += shares * adj_close_price
                shares = 0
        
        # Update portfolio value
        if shares > 0 and adj_close_price > 0:
             portfolio_values.iloc[i] = shares * adj_close_price
        else: 
             portfolio_values.iloc[i] = capital

    # Liquidate at the end if still holding shares
    if shares > 0 and not data.empty and data['Adj Close'].iloc[-1] > 0: 
        capital += shares * data['Adj Close'].iloc[-1]
        shares = 0
    
    if not data.empty: 
        portfolio_values.iloc[-1] = capital 
    elif not portfolio_values.empty: # Should not happen if data is empty, but as a safeguard
        portfolio_values.iloc[-1] = initial_capital


    portfolio_values = portfolio_values.fillna(method='ffill').fillna(initial_capital)
    return portfolio_values, data 

def calculate_performance_metrics(portfolio_values, risk_free_rate_annual=RISK_FREE_RATE, num_days_in_data=None):
    """Calculates performance metrics for a strategy."""
    if portfolio_values is None or portfolio_values.empty or portfolio_values.iloc[0] == 0:
        return {
            "Total Return (%)": 0, "Annualized Return (%)": 0,
            "Sharpe Ratio": np.nan, "Max Drawdown (%)": 0, # Return NaN for Sharpe if no trades
            "CAGR (%)": 0, "Number of Trades": 0 
        }

    total_return = (portfolio_values.iloc[-1] / portfolio_values.iloc[0] - 1) * 100 if portfolio_values.iloc[0] !=0 else 0
    
    daily_returns = portfolio_values.pct_change().dropna()
    if daily_returns.empty or daily_returns.std() == 0: # Check for std dev = 0
         return {
            "Total Return (%)": total_return, "Annualized Return (%)": 0,
            "Sharpe Ratio": np.nan, "Max Drawdown (%)": 0, # Sharpe is undefined
            "CAGR (%)": 0, "Number of Trades": 0
        }

    annualized_return = ((1 + daily_returns.mean()) ** 252 - 1) * 100
    
    daily_risk_free_rate = (1 + risk_free_rate_annual)**(1/252) - 1
    excess_returns = daily_returns - daily_risk_free_rate
    sharpe_ratio = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252) if excess_returns.std() != 0 else np.nan
    
    cumulative_returns = (1 + daily_returns).cumprod()
    peak = cumulative_returns.expanding(min_periods=1).max()
    drawdown = (cumulative_returns / peak) - 1
    max_drawdown = drawdown.min() * 100
    
    if num_days_in_data is None and not portfolio_values.index.empty: 
        num_days_in_data = (portfolio_values.index[-1] - portfolio_values.index[0]).days
    elif num_days_in_data is None: 
        num_days_in_data = 0

    num_years = num_days_in_data / 365.25 if num_days_in_data is not None and num_days_in_data > 0 else 0
    cagr = ((portfolio_values.iloc[-1] / portfolio_values.iloc[0]) ** (1 / num_years) - 1) * 100 if num_years > 0 and portfolio_values.iloc[0] != 0 else 0
    
    return {
        "Total Return (%)": round(total_return, 2),
        "Annualized Return (%)": round(annualized_return, 2),
        "Sharpe Ratio": round(sharpe_ratio, 2) if not np.isnan(sharpe_ratio) else np.nan,
        "Max Drawdown (%)": round(max_drawdown, 2),
        "CAGR (%)": round(cagr, 2),
    }

# --- Specific Strategies ---
def run_ma_crossover_strategy(stock_data, params, initial_capital):
    """Runs Moving Average Crossover strategy."""
    df = stock_data.copy()
    short_window = params['short_window']
    long_window = params['long_window']

    df['short_ma'] = calculate_sma(df, short_window)
    df['long_ma'] = calculate_sma(df, long_window)
    
    df['signal'] = 0
    df.loc[df['short_ma'] > df['long_ma'], 'signal'] = 1
    df.loc[df['short_ma'] < df['long_ma'], 'signal'] = -1
        
    portfolio_values, df_with_pos = backtest_strategy(df.copy(), initial_capital)
    if portfolio_values is None: return None, None, None

    num_trades = (df_with_pos['position'].diff().fillna(0) == 1).sum()

    metrics = calculate_performance_metrics(portfolio_values, num_days_in_data=len(stock_data))
    metrics["Number of Trades"] = int(num_trades)
    return metrics, df_with_pos, portfolio_values

def run_macd_strategy(stock_data, params, initial_capital):
    """Runs MACD Crossover strategy."""
    df = stock_data.copy()
    fast_period = params.get('fast_period', 12)
    slow_period = params.get('slow_period', 26)
    signal_period = params.get('signal_period', 9)

    df['macd'], df['signal_line'], df['histogram'] = calculate_macd(df, fast_period, slow_period, signal_period)
    
    df['signal'] = 0
    df.loc[df['macd'] > df['signal_line'], 'signal'] = 1
    df.loc[df['macd'] < df['signal_line'], 'signal'] = -1
    
    portfolio_values, df_with_pos = backtest_strategy(df.copy(), initial_capital)
    if portfolio_values is None: return None, None, None
    
    num_trades = (df_with_pos['position'].diff().fillna(0) == 1).sum()
    metrics = calculate_performance_metrics(portfolio_values, num_days_in_data=len(stock_data))
    metrics["Number of Trades"] = int(num_trades)
    return metrics, df_with_pos, portfolio_values

def run_rsi_strategy(stock_data, params, initial_capital):
    """Runs RSI Overbought/Oversold strategy."""
    df = stock_data.copy()
    rsi_period = params.get('rsi_period', 14)
    overbought_level = params.get('overbought_level', 70)
    oversold_level = params.get('oversold_level', 30)

    df['rsi'] = calculate_rsi(df, window=rsi_period)
    
    df['signal'] = 0 
    df['rsi_prev'] = df['rsi'].shift(1)

    # Buy signal: RSI crosses above oversold
    df.loc[(df['rsi'] > oversold_level) & (df['rsi_prev'] <= oversold_level), 'signal'] = 1
    # Sell signal: RSI crosses below overbought OR if RSI is simply in overbought (more aggressive exit)
    df.loc[(df['rsi'] < overbought_level) & (df['rsi_prev'] >= overbought_level), 'signal'] = -1 
    # If RSI is already deep in overbought, might be a signal to be out/sell
    df.loc[df['rsi'] > overbought_level, 'signal'] = -1 # Signal to be flat if overbought
    
    portfolio_values, df_with_pos = backtest_strategy(df.copy(), initial_capital)
    if portfolio_values is None: return None, None, None

    num_trades = (df_with_pos['position'].diff().fillna(0) == 1).sum()
    metrics = calculate_performance_metrics(portfolio_values, num_days_in_data=len(stock_data))
    metrics["Number of Trades"] = int(num_trades)
    return metrics, df_with_pos, portfolio_values

def run_custom_supertrend_strategy(stock_data, params, initial_capital):
    """
    Custom SuperTrend strategy using clustering to select the optimal factor.
    Adapted from Reference.py for use in the backtesting framework.
    """
    import numpy as np
    import pandas as pd
    df = stock_data.copy()
    if 'High' not in df.columns or 'Low' not in df.columns or 'Close' not in df.columns:
        return None, None, None
    ATR_LENGTH = params.get('atr_length', 10)
    MIN_MULT = params.get('min_mult', 1.0)
    MAX_MULT = params.get('max_mult', 5.0)
    STEP = params.get('step', 0.5)
    PERF_ALPHA = params.get('perf_alpha', 10)
    FROM_CLUSTER = params.get('from_cluster', 'Best')
    MAX_ITER = params.get('max_iter', 1000)
    df['hl2'] = (df['High'] + df['Low']) / 2
    df['prev_close'] = df['Close'].shift(1)
    df['tr1'] = df['High'] - df['Low']
    df['tr2'] = (df['High'] - df['prev_close']).abs()
    df['tr3'] = (df['Low'] - df['prev_close']).abs()
    df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    df['atr'] = df['tr'].ewm(alpha=2/(ATR_LENGTH+1), adjust=False).mean()
    df = df.dropna().reset_index(drop=False)
    n = len(df)
    def sign(x):
        return np.where(x>0, 1, np.where(x<0, -1, 0))
    def compute_supertrend(df, factor, perf_alpha):
        arr_close = df['Close'].values
        arr_hl2 = df['hl2'].values
        arr_atr = df['atr'].values
        trend = np.zeros(n, dtype=int)
        upper = np.zeros(n, dtype=float)
        lower = np.zeros(n, dtype=float)
        output = np.zeros(n, dtype=float)
        perf = np.zeros(n, dtype=float)
        trend[0] = 1 if arr_close[0] > arr_hl2[0] else 0
        upper[0] = arr_hl2[0]
        lower[0] = arr_hl2[0]
        output[0] = arr_hl2[0]
        perf[0] = 0.0
        for i in range(1, n):
            up = arr_hl2[i] + arr_atr[i]*factor
            dn = arr_hl2[i] - arr_atr[i]*factor
            if arr_close[i] > upper[i-1]:
                trend[i] = 1
            elif arr_close[i] < lower[i-1]:
                trend[i] = 0
            else:
                trend[i] = trend[i-1]
            if arr_close[i-1] < upper[i-1]:
                upper[i] = min(up, upper[i-1])
            else:
                upper[i] = up
            if arr_close[i-1] > lower[i-1]:
                lower[i] = max(dn, lower[i-1])
            else:
                lower[i] = dn
            diff_sign = sign(arr_close[i-1] - output[i-1])
            perf[i] = perf[i-1] + 2/(perf_alpha+1)*((arr_close[i] - arr_close[i-1]) * diff_sign - perf[i-1])
            output[i] = lower[i] if trend[i] == 1 else upper[i]
        return {
            'trend': trend,
            'upper': upper,
            'lower': lower,
            'output': output,
            'perf': perf,
            'factor': factor
        }
    factors = np.arange(MIN_MULT, MAX_MULT + 0.0001, STEP)
    st_results = [compute_supertrend(df, f, PERF_ALPHA) for f in factors]
    perf_vals = np.array([res['perf'][-1] for res in st_results])
    fact_vals = np.array([res['factor'] for res in st_results])
    def k_means(data, factors, k=3, max_iter=1000):
        c1, c2, c3 = np.percentile(data, [25, 50, 75])
        centroids = np.array([c1, c2, c3])
        for _ in range(max_iter):
            clusters = {0: [], 1: [], 2: []}
            cluster_factors = {0: [], 1: [], 2: []}
            for d, f in zip(data, factors):
                dist = np.abs(d - centroids)
                idx = dist.argmin()
                clusters[idx].append(d)
                cluster_factors[idx].append(f)
            new_centroids = np.array([np.mean(clusters[i]) if len(clusters[i])>0 else centroids[i] for i in range(k)])
            if np.allclose(new_centroids, centroids):
                break
            centroids = new_centroids
        return clusters, cluster_factors, centroids
    clusters, cluster_factors, centroids = k_means(perf_vals, fact_vals, k=3, max_iter=MAX_ITER)
    order = np.argsort(centroids)
    sorted_clusters = {i: clusters[j] for i,j in enumerate(order)}
    sorted_cluster_factors = {i: cluster_factors[j] for i,j in enumerate(order)}
    if FROM_CLUSTER == 'Best':
        chosen_index = 2
    elif FROM_CLUSTER == 'Average':
        chosen_index = 1
    else:
        chosen_index = 0
    if len(sorted_cluster_factors[chosen_index])>0:
        target_factor = np.mean(sorted_cluster_factors[chosen_index])
    else:
        target_factor = factors[-1]
    st_final = compute_supertrend(df, target_factor, PERF_ALPHA)
    df['supertrend'] = st_final['output']
    df['signal'] = 0
    os = np.zeros(n, dtype=int)
    os[0] = 1 if df['Close'].iloc[0] > st_final['upper'][0] else 0
    for i in range(1, n):
        c = df['Close'].iloc[i]
        up = st_final['upper'][i]
        dn = st_final['lower'][i]
        if c > up:
            os[i] = 1
        elif c < dn:
            os[i] = 0
        else:
            os[i] = os[i-1]
    df['position'] = os
    df['signal'] = df['position'].diff().fillna(0)
    portfolio_values, df_with_pos = backtest_strategy(df, initial_capital)
    if portfolio_values is None:
        return None, None, None
    metrics = calculate_performance_metrics(portfolio_values, num_days_in_data=len(df))
    metrics['Number of Trades'] = (df_with_pos['position'].diff().fillna(0) == 1).sum()
    return metrics, df_with_pos, portfolio_values
