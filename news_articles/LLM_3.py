import pandas as pd
import transformers
from tqdm import tqdm
import csv
import torch
import ast
import numpy as np
import multiprocessing as mp
from transformers import AutoModelForCausalLM, AutoTokenizer
transformers.logging.set_verbosity_error()
torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
            # If the current chunk is not empty, save it and start a new chunk
            if current_chunk:
                chunks.append(tokenizer.convert_tokens_to_string(current_chunk))
            current_chunk = sentence_tokens
            current_length = sentence_length
       

    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append(tokenizer.convert_tokens_to_string(current_chunk))

    return chunks

df = pd.read_csv("AAPL_match.csv")

df = df.iloc[:]


prompt_template = (
    """<|im_start|>system
    You are an expert in analyzing stock market news. Please provide a sentiment score on how this news will influence the stock price, in terms of {aspect} which are related to the company. 
    On a scale of 1 to 10, 1 stands for very negative, 5 stands for neutral or not related, 10 stands for very positive. Only Provide the number! Remember, only provide a single integer ranging from 1 to 10 without anything else!
    Remember! The First and the only word you should output is a number from 1 to 10!
    <|im_end|>"""

    """<|im_start|>user
    {passage}
    <|im_end|>"""
    """<|im_start|>
    assistant"""
)

# prompt_template = (
#     """<|im_start|>system
#     You are an expert in analyzing the influence of stock market news on stock prices. Please review the following news passage and provide a sentiment score for each mentioned aspect in terms of its potential impact on the stock price. The aspects to consider are: profitability, market share, leadership, product innovation, and customer satisfaction.

#     For each aspect, provide a score on a scale of 1 to 10, where 1 stands for a very negative impact, 5 stands for neutral or unrelated, and 10 stands for a very positive impact. Provide your scores in the format "[Aspect]: [Score]", and ensure that each score is a single integer ranging from 1 to 10 without any additional text or commentary.

#     Example:
# Profitability: 4,
# Market Share: 6,
# Leadership: 5,
# Product Innovation: 8,
# Customer Satisfaction: 7,

#     Remember, your response should only contain the aspects and their corresponding scores as shown in the example above. Please review the following news excerpt and provide your analysis:
#     <|im_end|>"""

#     """<|im_start|>user
#     {passage}
#     <|im_end|>"""
#     """<|im_start|>
#     assistant"""
# )

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

tokenizer = AutoTokenizer.from_pretrained("teknium/OpenHermes-2.5-Mistral-7B")

model_id="teknium/OpenHermes-2.5-Mistral-7B"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    torch_dtype=torch_dtype,
    device_map='auto' if torch.cuda.is_available() else None
)

model.eval()
model.to(device)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model.config.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)

def get_model_responses(messages):
    # Encode the messages and get the attention masks
    encodings = tokenizer(messages, return_tensors="pt", padding=True, truncation=True)#, max_length=model.config.n_positions
    input_ids = encodings['input_ids'].to(device)
    attention_mask = encodings['attention_mask'].to(device)
    gen_config = {
    "temperature": 0.1,
    "top_p": 0.99,
    "repetition_penalty": 0.01,
    "top_k": 50,
	"num_return_sequences": 1,
	"eos_token_id": tokenizer.eos_token_id,
	"max_new_tokens": 3,
    }
    # Generate responses in batch
    outputs = model.generate(
        input_ids,
        attention_mask=attention_mask,
        # max_length=4096,  # or any other generation parameters you want to set
        do_sample=True,
        **gen_config,
    )
    
    # Decode all responses
    responses = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    return responses




# New CSV filename
output_csv = "AAPL_sentiment_analysis.csv"

# Analyze sentiment for each article and aspect

# prompts=[]


# def process_row(args):
#     index, row, max_tokens = args
#     prompts = []
#     article_text = row["article_text"]
#     aspects = ast.literal_eval(row["matched_names"])
#     passages = split_into_token_chunks(article_text, tokenizer, max_tokens)
#     for aspect in aspects:
#         for passage in passages:
#             prompt = prompt_template.format(aspect, aspect, passage)
#             prompts.append(prompt)
#     return prompts

# # Function to initialize tokenizer in worker processes
# def init_worker():
#     global tokenizer

# # Prepare data for multiprocessing
# data_for_mp = [(index, row, 1800) for index, row in df.iterrows()]

# # Initialize multiprocessing.Pool() with the desired number of processes
# pool = mp.Pool(processes=mp.cpu_count(), initializer=init_worker)

# # Use `imap` with `tqdm` to process data in parallel and display progress
# results = []
# for result in tqdm(pool.imap(process_row, data_for_mp, chunksize=10), total=len(data_for_mp), desc="Generating prompts"):
#     results.append(result)

# # Close the pool and wait for the work to finish
# pool.close()
# pool.join()

# # Flatten the results and update prompts_count as needed
# prompts_count = np.cumsum([len(sublist) for sublist in results])

# prompts = [item for sublist in results for item in sublist]
# print(len(prompts))


# print(prompts_count)
# prompts_generator=(p for p in prompts)
# response = []
# row_count = 0

with open(output_csv, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(["url", "aspect", "sentiment_output"])

    with torch.no_grad():
        prompts_count=np.zeros(df.shape[0],dtype = int)
        for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="generating prompt"):
            responses=[]
            article_text = row["article_text"]
            aspects = ast.literal_eval(row["matched_names"])
            passages = split_into_token_chunks(article_text, tokenizer, 3900) 
            row_prompts_num = 0
            for aspect in aspects:
                for passage in passages:
                    prompt = prompt_template.format(aspect=aspect, passage=passage)
                    # print(prompt)
                    response = get_model_responses([prompt])
                    # answers = [n.split('assistant')[-1].strip() for n in response]
                    # print(answers)
                    responses.append(response[0].split('assistant')[-1].strip())
            # print(aspects)
            # print(f"prompts_needed:{len(aspects)*len(passages)} actual_prompts:{len(responses)}")
            url=row['url']
            writer.writerow([url, aspects, responses])
            file.flush()

            
            

        # # Create a batch of prompts to feed to the model
        # prompt_batch = []
        # for prompt in prompts:
        #     prompt_batch.append(prompt)
        #     # When the batch size is 2, process the prompts through the model
        #     if len(prompt_batch) == 2:
        #         batch_responses = get_model_responses(prompt_batch)
        #         # Process the responses and write to CSV
        #         for response in batch_responses:
        #             # Extract the relevant data for writing to CSV
        #             response = response.split("<|im_start|>assistant")[-1].strip()
        #             # Here, you would determine the corresponding URL and aspect for each response
        #             # This is placeholder logic; adjust according to your actual data structure
        #             url = "some_url"
        #             aspect = "some_aspect"
        #             writer.writerow([url, aspect, response])
        #             file.flush()
        #         # Clear the batch to start a new one
        #         prompt_batch = []

        # # Don't forget to process the last batch if it's not empty
        # if prompt_batch:
        #     batch_responses = get_model_responses(prompt_batch)
        #     for response in batch_responses:
        #         response = response.split("<|im_start|>assistant")[-1].strip()
        #         url = "some_url"
        #         aspect = "some_aspect"
        #         writer.writerow([url, aspect, response])
        #         file.flush()


# with open(output_csv, mode='w', newline='', encoding='utf-8') as file:
#     writer = csv.writer(file)
#     writer.writerow(["url", "aspect", "sentiment_output"])#, "passage"
#     with torch.no_grad():
#         for i, prompts in tqdm(enumerate(prompts_generator),total=len(prompts),desc='analysing text'):#
#             output = out[0]["generated_text"][len(prompts[i]):].strip()
#             response.append(output)
#             # print(prompts[i+1])
#             if i == prompts_count[row_count]-1:
#                 row=df.iloc[row_count]
#                 aspect = row['matched_names']
#                 url = row["url"]
#                 writer.writerow([url, aspect, response])
#                 file.flush()
#                 response = []
#                 row_count += 1
#                 # torch.cuda.empty_cache()

