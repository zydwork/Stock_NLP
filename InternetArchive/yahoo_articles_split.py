import pandas as pd

# Load the large CSV file
df = pd.read_csv('yahoo_links_1.csv')

df_shuffled = df.sample(frac=1).reset_index(drop=True)

# Calculate the length of each split
total_rows = len(df)
ratio_sum = 4 + 3 + 2
len_part1 = total_rows * 4 // ratio_sum
len_part2 = total_rows * 3 // ratio_sum
# The last part will take the remaining rows

# Split the DataFrame
part1 = df.iloc[:len_part1]
part2 = df.iloc[len_part1:len_part1 + len_part2]
part3 = df.iloc[len_part1 + len_part2:]

# Save the splits to new CSV files
part1.to_csv('yahoo_links_a.csv', index=False)
part2.to_csv('yahoo_links_b.csv', index=False)
part3.to_csv('yahoo_links_c.csv', index=False)