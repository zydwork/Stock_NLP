import pandas as pd
import transformers
from tqdm import tqdm
import csv
import torch
import ast
import numpy as np
import multiprocessing as mp

torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
# torch.set_num_threads(36)
# Function to split text into chunks of sentences
def split_into_chunks(text, min_len, max_len):
    sentences = text.split('.')
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 < max_len:  # +1 for the space or period
            current_chunk += (sentence + '.') if sentence else ''
        else:
            if len(current_chunk) >= min_len:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + '.' if sentence else ''
    if len(current_chunk) >= min_len:
        chunks.append(current_chunk.strip())
    return chunks

def split_into_token_chunks(text, tokenizer, max_tokens):
    # Split text into sentences based on periods.
    sentences = text.split('.')
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        # Add the period back to each sentence except the last one
        if len(sentence) > 0:
            sentence += '.'

        # Tokenize the sentence
        sentence_tokens = tokenizer.tokenize(sentence)
        sentence_length = len(sentence_tokens)

        # Check if adding this sentence would exceed the max length
        if current_length + sentence_length <= max_tokens:
            # Add the sentence tokens to the current chunk
            current_chunk.extend(sentence_tokens)
            current_length += sentence_length
        else:
            print(current_length)
            # If the current chunk is not empty, save it and start a new chunk
            if current_chunk:
                chunks.append(tokenizer.convert_tokens_to_string(current_chunk))
            current_chunk = sentence_tokens
            current_length = sentence_length
       

    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append(tokenizer.convert_tokens_to_string(current_chunk))

    return chunks
# def split_into_token_chunks(text, tokenizer, max_tokens):
#     # Tokenize the text and identify sentence boundaries by periods.
#     tokens = tokenizer.tokenize(text)
#     chunks = []
#     current_chunk = []
#     current_length = 0

#     for token in tokens:
#         # Calculate the token length only once
#         token_length = len(tokenizer.tokenize(token)) if current_chunk else 0

#         # If the current token is a period, mark it as a sentence boundary
#         is_sentence_boundary = token == tokenizer.sep_token or token == "."


#         # If adding the token would exceed max_tokens, or it's a sentence boundary,
#         # we finalize the current chunk and start a new one
#         if current_length + token_length > max_tokens or (is_sentence_boundary and current_chunk):
#             # print((is_sentence_boundary and current_chunk))
#             print('token_count:',current_length + token_length, max_tokens) 
#             chunks.append(tokenizer.convert_tokens_to_string(current_chunk))
#             current_chunk = [token] if not is_sentence_boundary else []  # Start a new chunk
#             current_length = token_length if not is_sentence_boundary else 0

#         else:
#             # Add the token to the current chunk
#             current_chunk.append(token)
#             current_length += token_length
#     print(current_length)
#     # Don't forget to add the last chunk if it's not empty
#     if current_chunk:
#         chunks.append(tokenizer.convert_tokens_to_string(current_chunk))

#     return chunks
# Read the CSV file into a DataFrame
df = pd.read_csv("yahoo_matched_articles/AAPL_match.csv")

df = df.iloc[:2000]


prompt_template = (
    """<|im_start|>system
    You are an expert in analyzing stock market news. Did this part mention {}? How do you think the sentiment of {} from this part?
    <|im_end|>"""
    """
    On a scale of 1 to 10, 1 stands for very negative, 5 stands for neutral, 10 stands for very positive. Only Provide the number! Remember, only provide a single integer ranging from 1 to 10 without anything else!
    """
    """<|im_start|>user
    {}
    <|im_end|>"""
    """<|im_start|>assistant"""
)

# prompt_template = (
#     """
#     You are an expert in analyzing stock market news. Did this part mention {}? How do you think the sentiment of {} from this part?
#     """
#     """
#     On a scale of 1 to 10, 1 stands for very negative, 5 stands for neutral, 10 stands for very positive. Only Provide the number! Remember, only provide a single integer ranging from 1 to 10 without anything else! \n\n\n
#     """
#     """
#     {}
#     """
# )



model_path="teknium/OpenHermes-2.5-Mistral-7B"
pipeline = transformers.pipeline(
		"text-generation",
		model=model_path,
		torch_dtype=torch.bfloat16,
		device_map="cuda",
	)


pipeline.tokenizer.padding_side = 'left'
if pipeline.tokenizer.pad_token is None:
    pipeline.tokenizer.add_special_tokens({"pad_token":"<pad>"})
pipeline.model.resize_token_embeddings(len(pipeline.tokenizer))
pipeline.model.config.pad_token_id = pipeline.tokenizer.pad_token_id
pipeline.model.eval()

gen_config = {
    "temperature": 0.1,
    "top_p": 0.95,
    "repetition_penalty": 0.01,
    "top_k": 50,
	"do_sample": True,
	"num_return_sequences": 2,
	"eos_token_id": pipeline.tokenizer.eos_token_id,
	"max_new_tokens": 2,
    # "force_words_ids":['1','2','3','4','5','6','7','8','9','0']
}

pipeline.model.config.generation = gen_config


# New CSV filename
output_csv = "AAPL_sentiment_analysis.csv"

# Analyze sentiment for each article and aspect

prompts=[]
# prompts_count=np.zeros(df.shape[0],dtype = int)
# for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="generating prompt"):
#     article_text = row["article_text"]
#     aspects = ast.literal_eval(row["matched_names"])
#     passages = split_into_token_chunks(article_text, pipeline.tokenizer, 1800) 
#     row_prompts_num = 0
#     for aspect in aspects:
#         for passage in passages:
#             prompt = prompt_template.format(aspect, aspect, passage)
#             # print(prompt)
#             prompts.append(prompt)
#             row_prompts_num += 1
#     prompts_count[index]=prompts_count[index-1]+row_prompts_num

# Function to process a single row of the DataFrame
def process_row(args):
    index, row, max_tokens = args
    prompts = []
    article_text = row["article_text"]
    aspects = ast.literal_eval(row["matched_names"])
    passages = split_into_token_chunks(article_text, tokenizer, max_tokens)
    for aspect in aspects:
        for passage in passages:
            prompt = prompt_template.format(aspect, aspect, passage)
            prompts.append(prompt)
    return prompts

# Function to initialize tokenizer in worker processes
def init_worker():
    global tokenizer
    tokenizer = pipeline.tokenizer

# Prepare data for multiprocessing
data_for_mp = [(index, row, 1800) for index, row in df.iterrows()]

# Initialize multiprocessing.Pool() with the desired number of processes
pool = mp.Pool(processes=mp.cpu_count(), initializer=init_worker)

# Use `imap` with `tqdm` to process data in parallel and display progress
results = []
for result in tqdm(pool.imap(process_row, data_for_mp, chunksize=10), total=len(data_for_mp), desc="Generating prompts"):
    results.append(result)

# Close the pool and wait for the work to finish
pool.close()
pool.join()

# Flatten the results and update prompts_count as needed
prompts_count = np.cumsum([len(sublist) for sublist in results])

prompts = [item for sublist in results for item in sublist]
print(len(prompts))


print(prompts_count)
prompts_generator=(p for p in prompts)
response = []
row_count = 0


with open(output_csv, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(["url", "aspect", "sentiment_output"])#, "passage"
    with torch.no_grad():
        for i, out in tqdm(enumerate(pipeline(prompts_generator, batch_size=2, **gen_config)),total=len(prompts),desc='analysing text'):#
            output = out[0]["generated_text"][len(prompts[i]):].strip()
            response.append(output)
            # print(prompts[i+1])
            if i == prompts_count[row_count]-1:
                row=df.iloc[row_count]
                aspect = row['matched_names']
                url = row["url"]
                writer.writerow([url, aspect, response])
                file.flush()
                response = []
                row_count += 1
                # torch.cuda.empty_cache()

