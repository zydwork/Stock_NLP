import pandas as pd
import os
from copy import deepcopy

# Your directory
direc = '240M_Nor240'

# Get the list of folders (dates)
folders = os.listdir(direc)
folders = [folder for folder in folders if os.path.isdir(direc+'/'+folder) and folder.isdigit() and len(folder) == 8]

# Sort the folders in ascending order
folders.sort()

# Create a new directory to store the CSV files
os.makedirs('csv03', exist_ok=True)

# Set to store unique stock symbols
unique_symbols = set()

# Dictionary to store DataFrames for each symbol
symbol_data = {}

i=0
# Iterate over the folders
for folder in folders:
    print(folder)
    # Read the Parquet file
    data = pd.read_parquet(direc + '/' + folder + '/150000000.pqt')

    # Get the list of unique symbols in the current file
    symbols = data['symbol'].unique()

    # Add these symbols to the set of unique symbols
    unique_symbols.update(symbols)

    # Iterate over the symbols
    for symbol in symbols:
        # Create a DataFrame for the current symbol
        df_symbol = data[data['symbol'] == symbol]

        # If the symbol is already in the dictionary, append the new data. If it's not, add it to the dictionary.
        if symbol in symbol_data:
            symbol_data[symbol] = pd.concat([symbol_data[symbol], df_symbol], ignore_index=True)
            # symbol_data[symbol] = deepcopy(symbol_data[symbol])
        else:
            symbol_data[symbol] = df_symbol
    if not i%20:
        symbol_data=deepcopy(symbol_data)
        

# Convert the set of unique symbols to a DataFrame
df_symbols = pd.DataFrame(list(unique_symbols), columns=['symbol'])

# Save the DataFrame as a CSV file
df_symbols.to_csv(direc + 'csv03/unique_symbols.csv', index=False)

i=0
# Write each DataFrame in the dictionary to a CSV file
for symbol, df_symbol in symbol_data.items():
    print(i,symbol)
    df_symbol.to_csv(direc + 'csv03/' + symbol + '.csv', index=False)
    i+=1