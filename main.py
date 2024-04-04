from transformers import BertTokenizer, TFBertForMaskedLM
import tensorflow as tf

tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model = TFBertForMaskedLM.from_pretrained('bert-base-chinese')

text = "今天我[MASK]高興見到你"
masked_index = text.index("[MASK]")

input = tokenizer._encode_plus(text, return_tensors="tf")

prediction = model(input['input_ids'])

# predicted_index = tf.argmax(prediction.logits[0, masked_index]).numpy()
top_k_value, top_k_indices = tf.nn.top_k(prediction.logits[0, masked_index], k=3)
predicted_token = tokenizer.convert_ids_to_tokens(top_k_indices.numpy())

print(text.replace('[MASK]', predicted_token[0]))
print(text.replace('[MASK]', predicted_token[1]))
print(text.replace('[MASK]', predicted_token[2]))
