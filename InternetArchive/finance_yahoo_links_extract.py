import pandas as pd
import re

# Define input and output filenames
input_filename = 'finance_yahoo_news_raw.txt'
output_filename = 'finance_yahoo_links.csv'

# Read the file into a Pandas DataFrame
# Assuming the data doesn't contain any quoted delimiters, which would require setting `quoting=csv.QUOTE_NONE`
df = pd.read_csv(input_filename, delimiter=' ', header=None, usecols=[1,2], names=['date_time','url'])

print(df)
# Filter rows where the URL contains '.html'
df = df[df['url'].str.contains('.html')]

# Truncate the URLs at '.html'
df['url'] = df['url'].str.split('.html').str[0] + '.html'
df['url'] = df['url'].str.replace(':80', '', regex=False)
df['url'] = df['url'].str.replace('http:', 'https:', regex=False)

# #Remove the specified pattern "%20 ... 2525252F"
# pattern_to_remove = r'%20.*2F(?=.*2F)'
# df['url'] = df['url'].apply(lambda x: re.sub(pattern_to_remove, '', x, flags=re.DOTALL | re.IGNORECASE))

# Remove URLs containing "news/%"
df = df[~df['url'].str.contains('news/%')]
df = df[~df['url'].str.contains("news/'")]

# Remove duplicates
df.drop_duplicates(subset=['url'],inplace=True)
print(df)
# Write the unique URLs to a CSV file
df.to_csv(output_filename, index=False)

print(f"Unique URLs have been written to {output_filename}.")