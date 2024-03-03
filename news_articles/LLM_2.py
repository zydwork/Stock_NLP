# pip install bitsandbytes accelerate
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
hf_token = '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'
quantization_config = BitsAndBytesConfig(load_in_8bit=False)

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b", use_auth_token=hf_token)
model = AutoModelForCausalLM.from_pretrained("google/gemma-2b", quantization_config=quantization_config, use_auth_token=hf_token)

input_text = "Write me a poem about Machine Learning. Make it as longer as possible"
input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

outputs = model.generate(**input_ids)
print(tokenizer.decode(outputs[0]))
