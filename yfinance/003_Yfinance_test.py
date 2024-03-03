import yfinance as yf
import pandas as pd
pd.options.display.max_rows = 9999

ticker = yf.Ticker("AAPL")

print(ticker.info['longName'])