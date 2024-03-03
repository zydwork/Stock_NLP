import glob
import pandas as pd
import os

folder_path = 'investing_links_1'

# Use glob to find all CSV files in the folder
csv_files = glob.glob(os.path.join(folder_path, '*.csv'))

# Initialize an empty list to store DataFrames
dfs = []

df = pd.read_csv('investing_links.csv')

dfs.append(df)

# Loop over the list of csv files
for csv_file in csv_files:
    # Read the current CSV file into a DataFrame
    df = pd.read_csv(csv_file)
    
    # Append the DataFrame to the list
    dfs.append(df)


# Concatenate all DataFrames into a single DataFrame
merged_df = pd.concat(dfs, ignore_index=True)


merged_df.drop_duplicates(subset=['url'],inplace=True)
print(merged_df)
# (Optional) Save the merged DataFrame to a new CSV file
output_file = 'investing_links_1.csv'
merged_df.to_csv(output_file, index=False)


# # Calculate the length of each split
# total_rows = len(merged_df)
# ratio_sum = 4 + 3 + 2
# len_part1 = total_rows * 4 // ratio_sum
# len_part2 = total_rows * 3 // ratio_sum
# # The last part will take the remaining rows

# # Split the DataFrame
# part1 = merged_df.iloc[:len_part1]
# part2 = merged_df.iloc[len_part1:len_part1 + len_part2]
# part3 = merged_df.iloc[len_part1 + len_part2:]

# # Save the splits to new CSV files
# part1.to_csv('investing_links_a.csv', index=False)
# part2.to_csv('investing_links_b.csv', index=False)
# part3.to_csv('investing_links_c.csv', index=False)