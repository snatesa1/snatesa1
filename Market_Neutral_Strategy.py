from Yahoo_stock_utilites import calculate_profit, load_ticker_prices_ts_df, plot_strategy, load_ticker_ts_df
from modules import *
from Primal_functions import *
os.chdir(r'c:\Users\sathi\Github\snatesa1')
print (os.getcwd())

def cum_return():
    START_DATE = "2022-01-01"
    END_DATE = "2024-01-24"
    # tickers = ["AAPL", "MSFT", "AMZN", "TSLA", "GOOG", "XOM", "META", "NVDA", "PEP", "AVGO", "ADBE", "COST", 
            #    "PYPL", "AMD", "QCOM", "INTC", "TXN", "CHTR", "TMUS", "ISRG", "SBUX", "AMGN", "INTU", "ADP", "CSX", 
            #    "ADI", "MU", "ZM", "MAR", "GILD", "MELI", "WDAY", "PG", "PANW", "REGN", "RCL", "BKNG", "JNJ", "ADSK", "KLAC", "BAC"]
    
    tickers = ["TSM","GWRE","U"]
    tickers_df = load_ticker_prices_ts_df(tickers, START_DATE, END_DATE)
    # tickers_df = add_column(tickers_df,4)
    print (tickers_df)
    tickers_rets_df = tickers_df.dropna(axis=1).pct_change()  # first % is NaN
    
    # 1+ to allow the cumulative product of returns over time, and -1 to remove it at the end.
    tickers_rets_df = (1 + tickers_rets_df).cumprod() - 1
    print (tickers_rets_df.head())
    
    # plt.figure(figsize=(11, 11))
    # for ticker in tickers_rets_df.columns:
    #     plt.plot(tickers_rets_df.index, tickers_rets_df[ticker] * 100.0, label=ticker)
    
    # plt.xlabel("Date (Year-Month)")
    # plt.ylabel("Cummulative Returns(%")
    # plt.legend()
    # plt.show()
    
cum_return()