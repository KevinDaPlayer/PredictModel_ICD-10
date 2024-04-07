from datasets import load_dataset, ClassLabel, Value
import pandas as pd
from transformers import AutoTokenizer
import numpy as np

# 指定 CSV 文件路径
csv_file_path = 'your_dataset.csv'

# 加载数据集
dataset = load_dataset('csv', data_files=csv_file_path)

# 查看数据集结构
dataset['train'].features

# 假设我们使用 "Long Description" 作为输入，"Code" 作为预测目标
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def preprocess_function(examples):
    # 编码文本
    result = tokenizer(examples['Long Description'], truncation=True, padding='max_length', max_length=512)
    # 编码标签（这里假设你的标签是分类任务的形式）
    result["labels"] = [label_list.index(code) for code in examples["Code"]]
    return result

# 假设你已经有了所有唯一标签的列表
label_list = dataset['train'].unique('Code')
label_list.sort()  # 确保标签列表有序
num_labels = len(label_list)

# 应用预处理
encoded_dataset = dataset.map(preprocess_function, batched=True)


from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_labels)


from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
  output_dir="./results",
  evaluation_strategy="epoch",
  learning_rate=2e-5,
  per_device_train_batch_size=8,
  per_device_eval_batch_size=8,
  num_train_epochs=3,
  weight_decay=0.01,
)

trainer = Trainer(
  model=model,
  args=training_args,
  train_dataset=encoded_dataset["train"],
  eval_dataset=encoded_dataset["test"],  # 确保你有一个测试集
  tokenizer=tokenizer,
)

trainer.train()

trainer.evaluate()
