import json
import os
import pandas as pd
from rapidfuzz import fuzz
from dateutil import parser

from dateutil.tz import tzutc
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
import torch

model_name = "bert-base-uncased" 
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
            ceos_periods = extract_time_periods(company.get('ceos', []))
            board_members_periods = extract_time_periods(company.get('board_members', []))
            # Add company info to the result
            result[ticker] = {
                'aliases': company.get('aliases', []),
                'products': company.get('products', []),
                'subsidiaries': company.get('subsidiaries', []),
                'owned_entities': company.get('owned_entities', []),
                'ceos': ceos_periods,
                'board_members': board_members_periods
            }
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



import multiprocessing as mp
import numpy as np

def append_to_csv(ticker, matched_names, article):
    output_file = f'matched_articles/{ticker}_match.csv'
    # Check if file exists and whether headers need to be written
    write_header = not os.path.exists(output_file)
    # Data to append
    data_to_append = {
        'matched_names': [', '.join(matched_names)],
        'url': article['url'],
        'date_time': article['date_time'],
        'article_text': article['article_text']
    }
    # Convert data to DataFrame
    df_to_append = pd.DataFrame([data_to_append])
    # Append to CSV file
    df_to_append.to_csv(output_file, mode='a', index=False, header=write_header)

# Function to process a chunk of the DataFrame
def process_chunk(chunk, processed_data):
        # Set up the device for PyTorch
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    # Load the tokenizer and model for each process
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    model.to(device)

    # Create a NER pipeline for each process
    ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, device=0 if device.type == "cuda" else -1)
    for index, row in tqdm(chunk.iterrows(), total=chunk.shape[0], desc="Processing"):
        article_text = row['article_text']
        article_date = parser.parse(row['date_time'])
        article_matches = {}  # Dictionary to hold all matches for this article

        # Use the NER pipeline to extract entities
        ner_results = ner_pipeline(article_text)

        # Get the list of entity words from the NER results
        entity_words = [result['word'] for result in ner_results]

        # Check each ticker's data for matches with the extracted entities
        for ticker, value in processed_data.items():
            matches = []
            for attribute, names in value.items():
                for name in names:
                    if attribute in ['ceos', 'board_members']:
                        start_date, end_date = names[name]
                        if not is_within_period(article_date, start_date, end_date):
                            continue
                    for entity in entity_words:
                        if fuzz.partial_ratio(entity, name) > 95:
                            matches.append(entity)
            # If there were any matches for this ticker, add them to the article_matches
            if matches:
                article_matches[ticker] = matches

        # Now write each set of matches to its respective CSV
        for ticker, matched_names in article_matches.items():
            append_to_csv(ticker, matched_names, row)

from multiprocessing import set_start_method


if __name__ == '__main__':
    #     # Ensure that the GPU is available and detected by PyTorch
    # if torch.cuda.is_available():
    #     device = torch.device("cuda")
    #     print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    # else:
    #     device = torch.device("cpu")
    #     print("Using CPU")

    # # Load the tokenizer and model from Hugging Face (replace 'bert-base-uncased' with your FinBERT NER model)
    # tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    # model = AutoModelForTokenClassification.from_pretrained("bert-base-uncased")
    # model.to(device)

    # # Create a NER pipeline using the model and tokenizer
    # ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, device=0 if device.type == "cuda" else -1)

    # Read and process JSON files
    folder_path = 'info'
    processed_data = read_and_process_json_files(folder_path)
    # Read and process JSON files
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass  # Start method has already been set
    
    folder_path = 'info'
    processed_data = read_and_process_json_files(folder_path)

    # Read the news CSV file
    news_df = pd.read_csv('investing_articles_cleaned.csv')

    # Determine the number of processes and split the DataFrame into chunks
    num_processes = 26
    # mp.cpu_count()
    chunks = np.array_split(news_df, num_processes)


    # Create a pool of processes
    pool = mp.Pool(processes=num_processes)

    with mp.Pool(processes=num_processes) as pool:
        # Use starmap to pass multiple arguments to the function being called
        results = list(tqdm(pool.starmap(process_chunk, [(chunk, processed_data) for chunk in chunks]), total=len(chunks)))
    # Close the pool and wait for the work to finish
    pool.close()
    pool.join()



