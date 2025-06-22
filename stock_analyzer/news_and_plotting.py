# news_and_plotting.py

import plotly.express as px
import plotly.graph_objects as go
import feedparser
from datetime import datetime
import pandas as pd

# --- Strategy Performance Plotting (Plotly Express) ---
def plot_strategy_performance(stock_data_with_signals, portfolio_values, ticker, strategy_name, params_str, fib_levels):
    """Plots stock price with signals, indicators, and equity curve using Plotly Express."""
    # Price and signals
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=stock_data_with_signals.index, y=stock_data_with_signals['Adj Close'],
                             mode='lines', name='Adj Close', line=dict(color='blue')))
    # Buy/Sell signals
    buy_signals = stock_data_with_signals[stock_data_with_signals['position'].diff().fillna(0) == 1]
    sell_signals = stock_data_with_signals[stock_data_with_signals['position'].diff().fillna(0) == -1]
    fig.add_trace(go.Scatter(x=buy_signals.index, y=stock_data_with_signals.loc[buy_signals.index, 'Adj Close'],
                             mode='markers', name='Buy Signal', marker=dict(symbol='triangle-up', color='green', size=10)))
    fig.add_trace(go.Scatter(x=sell_signals.index, y=stock_data_with_signals.loc[sell_signals.index, 'Adj Close'],
                             mode='markers', name='Sell Signal', marker=dict(symbol='triangle-down', color='red', size=10)))
    # MAs
    if 'short_ma' in stock_data_with_signals.columns and 'long_ma' in stock_data_with_signals.columns:
        fig.add_trace(go.Scatter(x=stock_data_with_signals.index, y=stock_data_with_signals['short_ma'],
                                 mode='lines', name='Short MA', line=dict(color='orange', dash='dash')))
        fig.add_trace(go.Scatter(x=stock_data_with_signals.index, y=stock_data_with_signals['long_ma'],
                                 mode='lines', name='Long MA', line=dict(color='purple', dash='dash')))
    # Fibonacci
    if fib_levels:
        for level_name, level_val in fib_levels.items():
            if 'High' not in level_name and 'Low' not in level_name:
                fig.add_hline(y=level_val, line_dash='dot', line_color='grey', annotation_text=f'Fib {level_name.split("%") [0]}%')
    fig.update_layout(title=f"{ticker} - {strategy_name} ({params_str})",
                      yaxis_title='Price', legend_title='Legend', height=600)
    # Portfolio value (equity curve)
    fig2 = px.line(x=portfolio_values.index, y=portfolio_values.values, labels={'x': 'Date', 'y': 'Portfolio Value'}, title='Equity Curve')
    fig2.update_traces(line_color='teal')
    fig2.update_layout(height=400)
    # Return both figures for Streamlit display
    return fig, fig2

# --- News Fetching ---
def fetch_news(stock_ticker, num_articles=3):
    """Fetches financial news headlines using Google News and Yahoo Finance. Returns a list of dicts."""
    google_news_query_base = "https://news.google.com/rss/search?hl=en-US&gl=US&ceid=US:en&q="
    feeds_config = []
    if stock_ticker:
        safe_stock_ticker = str(stock_ticker).replace(' ', '+')
        feeds_config.append({"name": f"Google News for {stock_ticker}",
                             "url": f"{google_news_query_base}{safe_stock_ticker}+stock+OR+shares+OR+news"})
    feeds_config.append({"name": "Google News - Top Business Headlines",
                         "url": f"{google_news_query_base}business+finance+market+news"})
    feeds_config.append({"name": "Yahoo Finance - Top Stories",
                         "url": "https://finance.yahoo.com/rss/topstories"})
    all_headlines = []
    max_total_articles_to_display = num_articles * len(feeds_config)
    for feed_info in feeds_config:
        feed_name = feed_info["name"]
        feed_url = feed_info["url"]
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
            parsed_feed = feedparser.parse(feed_url, agent=headers.get('User-Agent'))
            if not parsed_feed.entries:
                continue
            count_from_this_feed = 0
            for entry in parsed_feed.entries:
                title = entry.get("title", "N/A")
                link = entry.get("link", "N/A")
                published_date_parsed = entry.get("published_parsed")
                published_date = published_date_parsed if published_date_parsed else None
                article_source_name = entry.get("source", {}).get("title") if entry.get("source") else feed_name
                all_headlines.append({
                    "title": title, "link": link, "published": published_date,
                    "source": article_source_name
                })
                count_from_this_feed += 1
                if count_from_this_feed >= num_articles:
                    break
        except Exception:
            continue
    if not all_headlines:
        return []
    # Sort by published date if possible
    try:
        all_headlines.sort(key=lambda x: x['published'] if x['published'] else 0, reverse=True)
    except Exception:
        pass
    return all_headlines[:max_total_articles_to_display]

# --- Macro-Stock Visualization (Plotly Express) ---
def create_macro_stock_visualization(macro_data_dict, stock_data_df, ticker_symbol_str):
    """Creates visualization comparing Unemployment, CPI, and Stock performance using Plotly Express."""
    if not macro_data_dict or stock_data_df is None or stock_data_df.empty:
        return None
    unemployment = macro_data_dict.get("Unemployment Rate")
    cpi_yoy = macro_data_dict.get("CPI (YoY Change)")
    if unemployment is None and cpi_yoy is None:
        return None
    valid_macro_series = [s for s in [unemployment, cpi_yoy] if s is not None and not s.empty]
    if not valid_macro_series:
        return None
    start_date_macro = min(s.index.min() for s in valid_macro_series)
    end_date_macro = datetime.now()
    stock_filtered = stock_data_df[(stock_data_df.index >= start_date_macro) & (stock_data_df.index <= end_date_macro)].copy()
    if stock_filtered.empty:
        return None
    stock_normalized = ((stock_filtered['Adj Close'] / stock_filtered['Adj Close'].iloc[0]) - 1) * 100
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=stock_normalized.index, y=stock_normalized.values,
                             mode='lines', name=f'{ticker_symbol_str} (% change)', line=dict(color='dodgerblue', width=2)))
    if unemployment is not None and not unemployment.empty:
        fig.add_trace(go.Scatter(x=unemployment.index, y=unemployment.values,
                                 mode='lines', name='Unemployment Rate (%)', line=dict(color='crimson', dash='dash')))
    if cpi_yoy is not None and not cpi_yoy.empty:
        fig.add_trace(go.Scatter(x=cpi_yoy.index, y=cpi_yoy.values,
                                 mode='lines', name='CPI YoY Change (%)', line=dict(color='forestgreen', dash='dot')))
    fig.update_layout(title=f'Macroeconomic Indicators vs {ticker_symbol_str} Stock Performance',
                      xaxis_title='Date', yaxis_title='Normalized Change / Macro Value',
                      legend_title='Legend', height=600)
    return fig
