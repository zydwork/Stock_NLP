# import pandas as pd
# import spacy
# from multiprocessing import Pool

# # Function to process each chunk of DataFrame
# def process_chunk(chunk):
#     nlp = spacy.load("en_core_web_sm")
#     chunk['cleaned_article'] = chunk['article_text'].apply(lambda article: ' '.join(sent.text.strip() for sent in nlp(article).sents))
#     return chunk

# # Function to split DataFrame into chunks
# def split_dataframe(df, chunk_size):
#     chunks = []
#     num_chunks = len(df) // chunk_size + 1
#     for i in range(num_chunks):
#         chunks.append(df[i * chunk_size:(i + 1) * chunk_size])
#     return chunks

# # Load the CSV file
# df = pd.read_csv('yahoo_articles_year_by_year/yahoo_articles_2005.0.csv')

# # Define the number of partitions
# num_partitions = 100  # Number of partitions to split dataframe
# num_cores = 28  # Number of cores on your machine

# # Split the DataFrame into chunks
# chunks = split_dataframe(df, len(df) // num_partitions)

# # Create a multiprocessing Pool
# pool = Pool(num_cores)

# # Process each chunk in parallel
# df_list = pool.map(process_chunk, chunks)

# # Concatenate the processed chunks
# df_concatenated = pd.concat(df_list)

# # Clean up
# pool.close()
# pool.join()

# # Drop the original articles column
# df_concatenated.drop(columns=['article_text'], inplace=True)

# # Save the cleaned data back to a CSV file if needed
# df_concatenated.to_csv('yahoo_articles_year_by_year/yahoo_articles_2005.0_cleaned.csv', index=False)


import pandas as pd
from multiprocessing import Pool

# Define the clean_newlines function to be applied to each chunk
def clean_newlines(text):
    cleaned_text = ""
    for part in text.split('\n'):
        if len(part) == 0 or part[-1] == '.':
            cleaned_text += part
        else:
            cleaned_text += part + ' '
    return cleaned_text.strip()

# Function to process each chunk of DataFrame
def process_chunk(chunk):
    chunk['cleaned_article'] = chunk['article_text'].apply(clean_newlines)
    return chunk

# Read the CSV file in chunks
chunk_size = 100  # Adjust chunk size based on your memory capacity and file size
reader = pd.read_csv('yahoo_articles_year_by_year/yahoo_articles_2005.0.csv', chunksize=chunk_size)

# Pool for multiprocessing: Adjust the number of processes based on your machine's capabilities
pool = Pool(processes=28)  # Example for a quad-core machine

# Process the chunks in parallel using pool.map
cleaned_chunks = pool.map(process_chunk, reader)

# Concatenate the processed chunks back together
cleaned_df = pd.concat(cleaned_chunks)

# Drop the original articles column
cleaned_df.drop(columns=['article_text'], inplace=True)

# Save the cleaned data back to a CSV file
cleaned_df.to_csv('yahoo_articles_year_by_year/yahoo_articles_2005.0_cleaned.csv', index=False)

# Close the pool and wait for the work to finish
pool.close()
pool.join()