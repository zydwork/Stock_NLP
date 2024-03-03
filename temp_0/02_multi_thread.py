import pandas as pd
import os
from multiprocessing import Pool, cpu_count
from copy import deepcopy

# Your directory
direc = '240M_Nor240'

# Get the list of folders (dates)
folders = os.listdir(direc)
folders = [folder for folder in folders if os.path.isdir(direc+'/'+folder) and folder.isdigit() and len(folder) == 8]

# Sort the folders in ascending order
folders.sort()

# Create a new directory to store the CSV files
os.makedirs('csv02', exist_ok=True)

# Set to store unique stock symbols
unique_symbols = set()

# Iterate over the folders
for folder in folders:
    print(folder)
    # Read the Parquet file
    data = pd.read_parquet(direc + '/' + folder + '/150000000.pqt')

    # Get the list of unique symbols in the current file
    symbols = data['symbol'].unique()

    # Add these symbols to the set of unique symbols
    unique_symbols.update(symbols)

# Convert the set of unique symbols to a DataFrame
df_symbols = pd.DataFrame(list(unique_symbols), columns=['symbol'])

# Save the DataFrame as a CSV file
df_symbols.to_csv('csv02/unique_symbols.csv', index=False)

def process_symbol(symbol):
    df_list = []
    for folder in folders:
        print(folder)
        # Read the Parquet file
        data = pd.read_parquet(direc + '/' + folder + '/150000000.pqt')

        # Create a DataFrame for the current symbol
        df_symbol = data[data['symbol'] == symbol]

        df_list.append(df_symbol)

    # Concatenate all DataFrames into one and save it as a CSV file
    df_symbol = pd.concat(df_list, ignore_index=True)
    df_symbol.to_csv('csv02/' + symbol + '.csv', index=False)

# Use a process pool to process multiple symbols simultaneously
with Pool(min(20, cpu_count())) as p:
    p.map(process_symbol, list(unique_symbols))