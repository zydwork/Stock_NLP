# Import the yfinance. If you get module not found error the run !pip install yfinance from your Jupyter notebook
import yfinance as yf

import pandas as pd



# Get the data for the stock AAPL
#data = yf.download('AAPL','2023-09-01', '2023-09-29', interval='2m')

data = yf.download('^DJI', start='1900-01-01', end='2030-01-01', interval='1d')
data.to_csv('^DJI.csv')
# Assuming 'data' is your DataFrame and it's indexed by date
data.index = pd.to_datetime(data.index)

# data = data.between_time('9:30', '16:00')
# # Import the plotting library
# import matplotlib.pyplot as plt
# #matplotlib inline

# # Calculate the 60-minute moving average
# #data_moving_avg = data['Adj Close'].rolling(window=60).mean()

# # Plot the moving average data
# #data_moving_avg.plot()

# # Plot the close price of the AAPL
# data['Adj Close'].plot()
# plt.show()