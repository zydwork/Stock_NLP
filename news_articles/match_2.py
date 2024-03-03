import json
import os
import pandas as pd
from rapidfuzz import fuzz
from dateutil import parser

from dateutil.tz import tzutc
from tqdm import tqdm

import multiprocessing as mp
import numpy as np

import re

# Function to check if the date of the article is within the start and end date of a CEO or board member
def is_within_period(article_date, start_date, end_date):
    # Convert all datetimes to aware datetimes in UTC for comparison
    if article_date and article_date.tzinfo is None:
        article_date = article_date.replace(tzinfo=tzutc())
    if start_date and start_date.tzinfo is None:
        start_date = start_date.replace(tzinfo=tzutc())
    if end_date and end_date.tzinfo is None:
        end_date = end_date.replace(tzinfo=tzutc())
    
    if start_date and end_date:
        return start_date <= article_date <= end_date
    elif start_date:
        return start_date <= article_date
    elif end_date:
        return article_date <= end_date
    else:
        return True

# Function to extract time periods from strings
def extract_time_periods(names):
    time_periods = {}
    if isinstance(names, str):
        names = [names]    
    for name_info in names:
        name_parts = name_info.split(" (")
        name = name_parts[0].strip()
        start_date, end_date = None, None
        
        # Look for Start and End times within the parts
        for part in name_parts[1:]:
            if 'Start:' in part:
                start_str = part.replace("Start:", "").replace("T00:00:00Z)", "").strip()
                try:
                    start_date = parser.parse(start_str)
                except (ValueError, parser.ParserError):
                    start_date = None
            elif 'End:' in part:
                end_str = part.replace("End:", "").replace("T00:00:00Z)", "").strip()
                try:
                    end_date = parser.parse(end_str)
                except (ValueError, parser.ParserError):
                    end_date = None
        
        time_periods[name] = (start_date, end_date)
    return time_periods

# Function to process and filter the JSON data
def process_json_data(json_data):
    result = {}
    for company in json_data:
        if (len(json_data) >= 2 and 'United States of America' in company['country']) or len(json_data)<=1:
            ticker = company['ticker']
            print(ticker)
            # Extract time periods for CEOs and board members
            # Add company info to the result
            result[ticker] = {
                'id_label': extract_time_periods(company.get('id_label', [])),
                'ticker': extract_time_periods(company.get('ticker', [])),
                'aliases': extract_time_periods(company.get('aliases', [])),
                'products': extract_time_periods(company.get('products', [])),
                'subsidiaries': extract_time_periods(company.get('subsidiaries', [])),
                'owned_entities': extract_time_periods(company.get('owned_entities', [])),
                'ceos': extract_time_periods(company.get('ceos', [])),
                'board_members': extract_time_periods(company.get('board_members', [])),
            }
    print(result)
    return result

# Process all JSON files in the 'info' folder
def read_and_process_json_files(folder_path):
    all_processed_data = {}
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r') as file:
                json_data = json.load(file)
                processed_data = process_json_data(json_data)
                all_processed_data.update(processed_data)
    return all_processed_data

# Read and process JSON files
folder_path = 'info'
processed_data = read_and_process_json_files(folder_path)

# Append matched articles to the CSV file with all matched strings
def append_to_csv(source_name ,ticker, matched_names, article):
    output_file = f'{source_name}_matched_articles/{ticker}_match.csv'
    # Check if file exists and whether headers need to be written
    write_header = not os.path.exists(output_file)
    # Data to append
    data_to_append = {
        'matched_names': matched_names,
        'url': article['url'],
        'date_time': article['date_time'],
        'article_text': article['article_text']
    }
    # Convert data to DataFrame
    df_to_append = pd.DataFrame([data_to_append])
    # Append to CSV file
    df_to_append.to_csv(output_file, mode='a', index=False, header=write_header)

# Function to process a chunk of the DataFrame
def process_chunk(source_name, chunk, processed_data):
    for index, row in tqdm(chunk.iterrows(), total=chunk.shape[0], desc="Processing"):
        article_text = str(row['article_text']) if row['article_text'] else ""
        article_date = parser.parse(row['date_time']) if row['date_time'] else None
        article_matches = {}  # Dictionary to hold all matches for this article
        def find_positions(pattern, text):
            return [match.start() for match in re.finditer(pattern, text)]

        # Check each ticker's data for matches
        for ticker, value in processed_data.items():
            matches = {}
            for attribute, names in value.items():
                for name, (start_date, end_date) in names.items():
                    if is_within_period(article_date, start_date, end_date):
                        if name.isupper():
                            if len(name)>1:
                                pattern = r'\b' + re.escape(name) + r'\b'                 
                                if re.search(pattern, article_text):
                                    matches[name] = find_positions(name, article_text)
                        elif not (name.islower() and name.replace(' ', '').isalpha()):
                            if fuzz.partial_ratio(article_text, name) > 95:
                                matches[name] = find_positions(name, article_text)
                        # elif fuzz.partial_ratio(article_text, name) > 95:
                        #     matches.append(name)                                                            

            # If there were any matches for this ticker, add them to the article_matches
            if matches:
                article_matches[ticker] = matches

        # Now write each set of matches to its respective CSV
        for ticker, matched_names in article_matches.items():
            # print(article_matches)
            append_to_csv(source_name, ticker, matched_names, row)

if __name__ == '__main__':
    # Read and process JSON files
    source_name = 'investing'
    folder_path = 'info'
    processed_data = read_and_process_json_files(folder_path)

    # Read the news CSV file
    news_df = pd.read_csv(f'{source_name}_articles_cleaned.csv')

    # Determine the number of processes and split the DataFrame into chunks
    num_processes = mp.cpu_count()*2
    chunks = np.array_split(news_df, num_processes)


    # Create a pool of processes
    pool = mp.Pool(processes=num_processes)

    with mp.Pool(processes=num_processes) as pool:
        # Use starmap to pass multiple arguments to the function being called
        results = list(tqdm(pool.starmap(process_chunk, [(source_name, chunk, processed_data) for chunk in chunks]), total=len(chunks)))
    # Close the pool and wait for the work to finish
    pool.close()
    pool.join()


# # Append matched articles to the CSV file
# def append_to_csv(ticker, matched_name, article):
#     output_file = f'matched_articles/{ticker}_match.csv'
#     # Check if file exists and whether headers need to be written
#     write_header = not os.path.exists(output_file)
#     # Data to append
#     data_to_append = {
#         'matched_name': [matched_name],
#         'url': [article['url']],
#         'date_time': [article['date_time']],
#         'article_text': [article['article_text']]
#     }
#     # Convert data to DataFrame
#     df_to_append = pd.DataFrame(data_to_append)
#     # Append to CSV file
#     df_to_append.to_csv(output_file, mode='a', index=False, header=write_header)

# # Iterate over the news DataFrame and match with tickers
# for index, row in news_df.iterrows():
#     print(index)
#     article_text = row['article_text']
#     article_date = parser.parse(row['date_time'])
#     for ticker, value in processed_data.items():
#         for attribute, names in value.items():
#             if attribute in ['ceos', 'board_members']:  # These have time periods
#                 for name, (start_date, end_date) in names.items():
#                     if is_within_period(article_date, start_date, end_date):
#                         if fuzz.partial_ratio(article_text, name) > 95: 
#                             append_to_csv(ticker, name, row)
#             else:  # 'aliases', 'products', 'subsidiaries', 'owned_entities'
#                 for name in names:
#                     if fuzz.partial_ratio(article_text, name) > 95:  # Assuming a threshold score of 70
#                         append_to_csv(ticker, name, row)
    


# ... [Keep the previously defined functions] ...


# # Function to process a chunk of the DataFrame
# def process_chunk(chunk, processed_data):
#     for index, row in tqdm(chunk.iterrows(), total=chunk.shape[0], desc="Processing"):
#         article_text = row['article_text']
#         article_date = parser.parse(row['date_time'])
#         for ticker, value in processed_data.items():
#             for attribute, names in value.items():
#                 if attribute in ['ceos', 'board_members']:  # These have time periods
#                     for name, (start_date, end_date) in names.items():
#                         if is_within_period(article_date, start_date, end_date):
#                             if fuzz.partial_ratio(article_text, name) > 95: 
#                                 append_to_csv(ticker, name, row)
#                 else:  # 'aliases', 'products', 'subsidiaries', 'owned_entities'
#                     for name in names:
#                         if fuzz.partial_ratio(article_text, name) > 95:  # Assuming a threshold score of 70
#                             append_to_csv(ticker, name, row)


