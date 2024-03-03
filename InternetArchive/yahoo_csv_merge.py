import glob
import pandas as pd
import os

folder_paths = ['yahoo_articles_csv_a','yahoo_articles_csv_b','yahoo_articles_csv_c']

dfs = []
for folder_path in folder_paths: 
    # Use glob to find all CSV files in the folder
    csv_files = glob.glob(os.path.join(folder_path, '*.csv'))

    # Initialize an empty list to store DataFrames
    

    # Loop over the list of csv files
    for csv_file in csv_files:
        print(csv_file)
        # Read the current CSV file into a DataFrame
        df = pd.read_csv(csv_file)
        
        # Append the DataFrame to the list
        dfs.append(df)

print('Merging dfs...')
merged_df = pd.concat(dfs, ignore_index=True)
print(merged_df)
print('Sorting by time ASC...')
merged_df.sort_values(by='date_time', ascending=True, inplace=True)
# (Optional) Save the merged DataFrame to a new CSV file
output_file = 'yahoo_articles_all.csv'
print(f'Writing file: {output_file}')
merged_df.to_csv(output_file, index=False)
