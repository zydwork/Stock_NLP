import pandas as pd
from pathlib import Path

# Define the path to your CSV file
input_csv_path = 'yahoo_articles_all.csv'

# Load the data into a pandas DataFrame
df = pd.read_csv(input_csv_path)
print(df)
print('splitting year')

# Ensure the date_time column is in datetime format
df['date_time'] = pd.to_datetime(df['date_time'], errors='coerce')

# Extract the year from the date_time column and create a new column for it
df['year'] = df['date_time'].dt.year

# Get a list of unique years in the DataFrame
years = df['year'].unique()

# Create an output directory if it doesn't exist
output_dir = Path('yahoo_articles_year_by_year')
output_dir.mkdir(parents=True, exist_ok=True)

# Loop through each unique year and create a separate CSV for each
for year in years:
    print(year)
    # Filter the DataFrame for the year
    df_year = df[df['year'] == year]
    
    # Define the output CSV file name based on the year
    output_csv_path = output_dir / f'yahoo_articles_{year}.csv'
    
    # Save the DataFrame for the current year to a CSV file
    df_year.to_csv(output_csv_path, index=False)

print("CSV files have been created for each year!")