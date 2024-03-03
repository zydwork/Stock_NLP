import pandas as pd
import os
from datetime import datetime
from dateutil import parser
from zoneinfo import ZoneInfo
import json

# Assuming 'path_to_json_folder' is the path to the folder containing JSON files
path_to_json_folder = 'yahoo_articles_c'
json_files = os.listdir(path_to_json_folder)
total_count=len(json_files)



def robust_parse(date_string):
    try:
        naive_dt = parser.parse(date_string)
                # Define Eastern timezone
        eastern = ZoneInfo('UTC')
        
        # Localize the naive datetime (attach the Eastern timezone information)
        localized_dt = naive_dt.replace(tzinfo=eastern)
        
        # Convert the localized datetime to UTC
        utc_dt = localized_dt.astimezone(ZoneInfo('UTC'))
        return utc_dt
    except (parser._parser.ParserError, ValueError) as e:
        # Handle the exception or return None
        print(f"Failed to parse date: {date_string}")
        return date_string

def load_json_file(json_path):
    encodings = ['utf-8', 'latin1', 'ISO-8859-1', 'windows-1252']  # List of encodings to try
    for encoding in encodings:
        try:
            with open(json_path, 'r', encoding=encoding) as f:
                data = json.load(f)
                df = pd.DataFrame([data])
                df['date_time'] = df['date_time'].apply(robust_parse)
                return df  # Return the data if successfully read
        except (UnicodeDecodeError, json.decoder.JSONDecodeError):
            pass  # If an error was encountered, try the next encoding
    print(f"Could not read file {json_path} with any of the specified encodings.")
    return None
# Loop through the list of json files

# Initialize an empty list to store DataFrames
df_list = []
local_count = 0
csv_count = 0
for json_file in json_files:
    json_path = os.path.join(path_to_json_folder, json_file)
    #print(json_path)
    # Read the JSON file into a DataFrame
    df = load_json_file(json_path)


    #df['date_time_updated'] = df['date_time_updated'].apply(robust_parse)
      # Note the list around `data`, this creates a single-row DataFrame
    df_list.append(df)
    
    local_count += 1
    # Append the DataFrame to the list
    total_count-=1
    
    if local_count>=10000:
        print(total_count)
        print(csv_count)
        merged_df = pd.concat(df_list, ignore_index=True)
        merged_df.to_csv(f'yahoo_articles_csv_c/yahoo_articles_c_{csv_count}.csv', index=False)
        csv_count+=1
        local_count=0
        df_list=[]

# Concatenate all DataFrames into one
merged_df = pd.concat(df_list, ignore_index=True)

# merged_df = pd.read_csv('investing_articles.csv')
# Sort the merged DataFrame by 'date_time' in ascending order
#merged_df.sort_values(by='date_time', ascending=True, inplace=True)
# # Then, when reading and converting dates in your DataFrame
# # merged_df['date_time'] = merged_df['date_time'].apply(robust_parse)
# # merged_df['date_time_updated'] = merged_df['date_time_updated'].apply(robust_parse)

# Export the large DataFrame to a CSV file
merged_df.to_csv(f'yahoo_articles_csv_c/yahoo_articles_c_{csv_count}.csv', index=False)