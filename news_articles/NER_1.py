# import pandas as pd
# from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
# import torch
# from tqdm import tqdm

# # Ensure GPU is available for PyTorch if you intend to use it, otherwise CPU will be used
# device = 0 if torch.cuda.is_available() else -1

# # Load the tokenizer and model from Hugging Face (replace with your chosen model)
# model_name = "dslim/bert-base-NER"  # This is an example model, replace with your chosen financial NER model
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForTokenClassification.from_pretrained(model_name)

# # Move model to GPU if available
# if device == 0:
#     model.cuda()

# # Create a NER pipeline using the model and tokenizer
# ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, device=device)

# def extract_named_entities(text, batch_size=16):
#     # Tokenize the text into a batch of inputs
#     inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt", max_length=512)
#     inputs = {key: val.to(device) for key, val in inputs.items()}

#     # Initialize an empty list to collect named entities
#     named_entities = []

#     # Process the batch of inputs
#     for i in range(0, inputs['input_ids'].size(0), batch_size):
#         batch = {key: val[i:i + batch_size] for key, val in inputs.items()}

#         # Get NER predictions for the batch
#         with torch.no_grad():
#             outputs = model(**batch)

#         # Decode the predictions
#         predictions = outputs.logits.argmax(dim=2)
#         tokens = tokenizer.convert_ids_to_tokens(batch['input_ids'][0])

#         # Iterate over token predictions
#         for token, pred in zip(tokens, predictions[0].tolist()):
#             # Map prediction index to entity string
#             entity = model.config.id2label[pred]
#             if entity in ['B-ORG', 'B-PER', 'B-LOC', 'B-MISC', 'I-PER', 'B-MISC','I-MISC','I-LOC','I-ORG']:  # Adjust entity types as needed
#                 named_entities.append(token)
#     print(named_entities)
#     # Return the list of named entities
#     return named_entities

# # You would call the function with the batch_size parameter when using it in `add_named_entities_column`
# def add_named_entities_column(input_csv, output_csv, batch_size=64):
#     # Read the CSV file
#     df = pd.read_csv(input_csv)

#     # Initialize a tqdm progress bar
#     tqdm.pandas(desc="Extracting Named Entities")

#     # Apply the NER in batches to each row in the "article text" column with progress bar
#     df['named_entities'] = df['article_text'].progress_apply(lambda text: extract_named_entities(text, batch_size=batch_size))

#     # Write the modified DataFrame to a new CSV file
#     df.to_csv(output_csv, index=False)


# # Replace 'input.csv' with the path to your CSV and 'output.csv' with the path for the output CSV
# input_csv = 'investing_articles_cleaned.csv'  # The CSV file containing financial news articles
# output_csv = 'investing_NER.csv'  # The CSV file where the output will be saved

# add_named_entities_column(input_csv, output_csv)

# import pandas as pd
# from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
# import torch
# from tqdm.auto import tqdm
# import os

# # Ensure GPU is available for PyTorch if you intend to use it, otherwise CPU will be used
# device = 0 if torch.cuda.is_available() else -1

# # Load the tokenizer and model from Hugging Face
# model_name = "dslim/bert-base-NER"  # Example model
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForTokenClassification.from_pretrained(model_name)

# # Move model to GPU if available
# if device == 0:
#     model.cuda()

# # Create a NER pipeline using the model and tokenizer
# ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, device=device)

# def extract_named_entities(text):
#     # Use the NER pipeline to extract entities
#     ner_results = ner_pipeline(text)
#     # Extract the entities and return them
#     entities = set()  # Use a set to avoid duplicates
#     for entity in ner_results:
#         if entity['entity'] in ['B-ORG', 'I-ORG', 'B-PER', 'I-PER', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']:
#             entities.add(entity['word'].replace('##', ''))
#     return list(entities)

# def process_article_and_append_to_csv(row, output_csv, batch_size=64):
#     # Extract named entities for the article text
#     named_entities = extract_named_entities(row['article_text'])
    
#     # Append the named entities to the row
#     row['named_entities'] = named_entities
    
#     # Append the row to the CSV file
#     with open(output_csv, 'a', newline='', encoding='utf-8') as f:
#         row.to_csv(f, header=f.tell()==0, index=False)

# # Replace 'input.csv' with the path to your CSV and 'output.csv' with the path for the output CSV
# input_csv = 'investing_articles_cleaned.csv'
# output_csv = 'investing_NER.csv'

# # Read the CSV file
# df = pd.read_csv(input_csv)

# # If the output CSV does not exist, create it with the header
# if not os.path.isfile(output_csv):
#     df.head(0).to_csv(output_csv, index=False)

# # Process each row and append to the CSV file
# for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Extracting Named Entities"):
#     process_article_and_append_to_csv(row, output_csv)

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from tqdm.auto import tqdm
import torch
import pandas as pd

# Ensure GPU is available for PyTorch if you intend to use it, otherwise CPU will be used
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the tokenizer and model from Hugging Face
model_name = "dslim/bert-base-NER"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name).to(device)

# Create a NER pipeline using the model and tokenizer
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, device=0 if device == "cuda" else -1)

# Function to extract named entities using the pipeline
def extract_named_entities(example):
    # Use the NER pipeline to extract entities
    ner_results = ner_pipeline(example["article_text"])
    # Extract the entities and return them
    batch_named_entities = []
    for result in ner_results:
        entities = set()
        for entity in ner_results:
            if entity['entity'] in ['B-ORG', 'I-ORG', 'B-PER', 'I-PER', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']:
                entities.add(entity['word'].replace('##', ''))
        batch_named_entities.append(list(entities))
    return {"named_entities": batch_named_entities}

# Load the CSV file into a Hugging Face Dataset
input_csv = 'investing_articles_cleaned.csv'
df = pd.read_csv(input_csv)
dataset = Dataset.from_pandas(df)

# Process the dataset to extract named entities
processed_dataset = dataset.map(extract_named_entities, batched=True)

# Convert the processed dataset back to a DataFrame
processed_df = processed_dataset.to_pandas()

# Write the DataFrame to a CSV file
output_csv = 'investing_NER.csv'
processed_df.to_csv(output_csv, index=False)