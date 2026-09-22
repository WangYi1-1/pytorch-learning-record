import os
import sys

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer
from transformers.utils import logging


sys.stdout.reconfigure(encoding="utf-8")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
logging.set_verbosity_error()


# 这是一个较小的预训练BERT：2层Transformer，每个token输出128维特征。
model_name = "google/bert_uncased_L-2_H-128_A-2"

instructions = [
    "turn left",
    "move forward and then turn right",
    "stop near the staircase"
]

# AutoTokenizer会自动加载与预训练BERT匹配的词表和分词规则。
tokenizer = AutoTokenizer.from_pretrained(model_name)

# AutoModel只加载BERT文本骨干网络，不带具体任务的分类头。
text_encoder = AutoModel.from_pretrained(model_name)

# tokenizer同时完成：分词、token ID转换、加特殊token、padding和attention mask。
encoded_inputs = tokenizer(
    instructions,
    padding=True,
    truncation=True,
    return_tensors="pt"
)

print("预训练模型：", model_name)
print("分词器类型：", type(tokenizer).__name__)
print("文本骨干网络类型：", type(text_encoder).__name__)
print("每个token的特征维度：", text_encoder.config.hidden_size)

print("\n每条指令的分词结果：")

for row_index, instruction in enumerate(instructions):
    token_ids = encoded_inputs["input_ids"][row_index]
    tokens = tokenizer.convert_ids_to_tokens(token_ids)

    print(f"{instruction:<35} → {tokens}")

print("\ninput_ids：")
print(encoded_inputs["input_ids"])
print("input_ids形状：", encoded_inputs["input_ids"].shape)

print("\nattention_mask：")
print(encoded_inputs["attention_mask"])
print(
    "attention_mask形状：",
    encoded_inputs["attention_mask"].shape
)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

text_encoder = text_encoder.to(device)

# BatchEncoding的.to(device)会把里面的所有Tensor一起送到CPU或GPU。
encoded_inputs = encoded_inputs.to(device)

text_encoder.eval()

with torch.no_grad():
    # **将字典拆成关键字参数，传给BERT的forward方法。
    outputs = text_encoder(**encoded_inputs)

# last_hidden_state保存每条指令、每个token的上下文特征。
token_features = outputs.last_hidden_state

# BERT在每句话开头自动加[CLS]，取第0个token作为句子特征。
sentence_features = token_features[:, 0, :]

print("\nBERT输出：")
print("token_features形状：", token_features.shape)
print("sentence_features形状：", sentence_features.shape)

# 新分类头是随机初始化的，将BERT的128维句子特征转换成4个动作分数。
classifier = nn.Linear(
    text_encoder.config.hidden_size,
    4
).to(device)

logits = classifier(sentence_features)

print("logits形状：", logits.shape)
print("\n未训练分类头的logits：")
print(logits)

total_parameters = sum(
    parameter.numel()
    for parameter in text_encoder.parameters()
)

print("\nBERT骨干网络参数量：", total_parameters)
print("使用设备：", device)
