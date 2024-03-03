# Import the necessary module
from yahoo_fin import stock_info as si

# Get a list of all S&P 500 tickers
sp500_tickers = si.tickers_nasdaq()

# Print the tickers
print(sp500_tickers)