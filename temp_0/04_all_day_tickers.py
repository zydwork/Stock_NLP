import pandas as pd
import os

# Your directory
direc = '240M_Nor240'

# Get the list of folders (dates)
folders = os.listdir(direc)
folders = [folder for folder in folders if os.path.isdir(direc+'/'+folder) and folder.isdigit() and len(folder) == 8]

# Sort the folders in ascending order
folders.sort()

# A dictionary to store the dates on which each ticker symbol appears
ticker_dates = {}

# Iterate over the folders
for folder in folders:
    print(folder)
    # Read the Parquet file
    data = pd.read_parquet(direc + '/' + folder + '/150000000.pqt')

    # Get the list of unique symbols in the current file
    symbols = data['symbol'].unique()

    # Update the dictionary
    for symbol in symbols:
        if symbol in ticker_dates:
            ticker_dates[symbol].add(folder)
        else:
            ticker_dates[symbol] = {folder}

# Find the ticker symbols that appear on all dates
all_date_tickers = [ticker for ticker, dates in ticker_dates.items() if len(dates) == len(folders)]

# Convert the list to a DataFrame
df_tickers = pd.DataFrame(all_date_tickers, columns=['symbol'])

# Save the DataFrame as a CSV file
df_tickers.to_csv('all_date_tickers.csv', index=False)