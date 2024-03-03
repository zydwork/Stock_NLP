import pandas as pd
import os

# Your directory
direc = '240M_Nor240'

# Get the list of folders (dates)
folders = os.listdir(direc)
folders = [folder for folder in folders if os.path.isdir(direc+'/'+folder) and folder.isdigit() and len(folder) == 8]

# Sort the folders in ascending order
folders.sort()

# Create a new directory to store the CSV files
os.makedirs('CSV_Files', exist_ok=True)

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

    # Iterate over the symbols
    for symbol in symbols:
        # Create a DataFrame for the current symbol
        df_symbol = data[data['symbol'] == symbol]

        # CSV file path
        csv_file = 'CSV_Files/' + symbol + '.csv'

        # If the CSV file exists, append the new data. If it doesn't exist, create a new file.
        if os.path.exists(csv_file):
            df_symbol.to_csv(csv_file, mode='a', header=False, index=False)
        else:
            df_symbol.to_csv(csv_file, index=False)

# Convert the set of unique symbols to a DataFrame
df_symbols = pd.DataFrame(list(unique_symbols), columns=['symbol'])

# Save the DataFrame as a CSV file
df_symbols.to_csv('CSV_Files/unique_symbols.csv', index=False)