from torch import bfloat16
import transformers



from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Ensure that your GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("teknium/OpenHermes-2.5-Mistral-7B")
# model = AutoModelForCausalLM.from_pretrained("teknium/OpenHermes-2.5-Mistral-7B")
# tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
# model = AutoModelForCausalLM.from_pretrained("google/gemma-2b")

model_id = "teknium/OpenHermes-2.5-Mistral-7B"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    torch_dtype=torch_dtype,
    device_map='auto' if torch.cuda.is_available() else None
)
model.eval()
model.to(device)

# Check if the tokenizer has a pad token, if not, set it to the eos token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Update the model's padding token id
model.config.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)

def get_model_response(message):
    # Encode the message and get the attention mask
    encoding = tokenizer(message, return_tensors="pt", padding=True, truncation=True)
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    # Generate a response
    outputs = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=300,  # Ensure this is a reasonable length for your GPU's memory capacity
        do_sample=True,
    )
    
    # Decode and print the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def get_model_responses(messages):
    # Encode the messages and get the attention masks
    encodings = tokenizer(messages, return_tensors="pt", padding=True, truncation=True)#, max_length=model.config.n_positions
    input_ids = encodings['input_ids'].to(device)
    attention_mask = encodings['attention_mask'].to(device)

    gen_config = {
    "temperature": 0.1,
    "top_p": 0.80,
    "repetition_penalty": 0.01,
    "top_k": 50,
	"num_return_sequences": 1,
	"eos_token_id": tokenizer.eos_token_id,
	"max_new_tokens": 30,
    }
    # Generate responses in batch
    outputs = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=4096,  # or any other generation parameters you want to set
        do_sample=True,
        **gen_config,
    )
    # Decode all responses
    responses = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    return responses

# Example message
message_1 ="""
    <|im_start|>system
    You are an expert in analyzing stock market news. Did this part mention apple? How do you think the sentiment of apple from this part?
    <|im_end|>
    On a scale of 1 to 10, 1 stands for very negative, 5 stands for neutral, 10 stands for very positive. Only Provide the number! Remember, only provide a single integer ranging from 1 to 10 without anything else!
    <|im_start|>user
    Apple launched its revolutionary vision pro. Outperforms its competitiors like meta.
    <|im_end|>
    <|im_start|>assistant<|im_end|>
    """

message_2 ="""
    <|im_start|>system
    You are an expert in analyzing stock market news. Did this part mention apple? How do you think the sentiment of apple from this part?
    <|im_end|>
    On a scale of 1 to 10, 1 stands for very negative, 5 stands for neutral, 10 stands for very positive. Only Provide the number! Remember, only provide a single integer ranging from 1 to 10 without anything else!
    <|im_start|>user
    Apple failed its electric car program ongoing for 10 years without any output.
    <|im_end|>
    <|im_start|>assistant<|im_end|>
    """

message_3 = """
    <|im_start|>system
    You are an expert in analyzing stock market news. Did this part mention Mac? How do you think the sentiment of Mac from this part?
    <|im_end|>
    On a scale of 1 to 10, 1 stands for very negative, 5 stands for neutral, 10 stands for very positive. Only Provide the number! Remember, only provide a single integer ranging from 1 to 10 without anything else!
    <|im_start|>user
    AlphaSmart Neo AlphaSmart has a new portable word processor out called the Neo which is basically a keyboard with a small 5. 75-inch by 1. 5-inch monochrome LCD screen attached to it.  It's aimed pretty much exclusively at the educational market, and is way cheaper and easier to carry than a laptop (at least it's easier to carry than most laptops) and more useful for taking notes or writing a term paper than a PDA (unless you invest in a good portable keyboard).  The major downside is that you basically still need a PC or Mac to transfer files to (via IrDA or a USB cable) if you want to print something out (unless you have an IrDA-enabled printer), but it does get up to 700 hours of battery life on just three AA batteries and retail for just $249.
    <|im_end|>
    <|im_start|>assistant
"""

# Get the model response
while True:
    responses = get_model_responses([message_3])
    answers = [response.split('<|im_start|>assistant')[-1].strip() for response in responses]
    print("\n###############\n")
    print(answers)