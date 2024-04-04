from transformers import AutoTokenizer, AutoModelForCausalLM
# import tensorflow as tf

model_name = "google/gemma-7b"

tokenizer = AutoTokenizer.from_pretrained("google/gemma-7b")
model = AutoModelForCausalLM.from_pretrained("google/gemma-7b").to("cuda")

input_text = "請問你了解ICD-10嗎"
input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

output = model.generate(**input_ids)
print(tokenizer.decode(output[0]))