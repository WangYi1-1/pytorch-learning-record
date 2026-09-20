import torch
from torch import nn

# 本课常用术语：
# 序列（sequence）：按顺序排列的一组token。
# 序列长度（sequence length）：一条指令中token的数量。
# padding：使用<pad>将较短句子补到相同长度。
# attention mask：标记真实token和padding位置。
# masked mean pooling：只对真实token求平均，忽略padding位置。


torch.manual_seed(42)

instructions = [
    "turn left",
    "move forward",
    "turn right and move forward",
    "stop"
]

vocabulary = {
    "<pad>": 0,
    "<unk>": 1,
    "turn": 2,
    "left": 3,
    "right": 4,
    "and": 5,
    "move": 6,
    "forward": 7,
    "stop": 8
}

pad_id = vocabulary["<pad>"]
unknown_id = vocabulary["<unk>"]

# 将每条文字指令转换成长度不同的token ID列表
encoded_instructions = []

for instruction in instructions:
    tokens = instruction.lower().split()
    token_ids = [
        vocabulary.get(token, unknown_id)
        for token in tokens
    ]

    # token_ids目前是Python列表，还不是Tensor，所以没有.shape属性。
    # 概念上的形状是[sequence_length]：
    # 例如"turn left"有2个token，可以看成[2]；
    # "turn right and move forward"有5个token，可以看成[5]。
    encoded_instructions.append(token_ids)

# 找到当前batch中最长指令的token数量
max_length = max(
    len(token_ids)
    for token_ids in encoded_instructions
)

padded_instructions = []
attention_masks = []

for token_ids in encoded_instructions:
    padding_length = max_length - len(token_ids)

    # 在真实token后面补<pad>，使每条指令长度都等于max_length
    padded_token_ids = (
        token_ids
        + [pad_id] * padding_length
    )

    # attention mask：
    # 1表示真实token，0表示补齐用的<pad>
    attention_mask = (
        [1] * len(token_ids)
        + [0] * padding_length
    )

    padded_instructions.append(padded_token_ids)
    attention_masks.append(attention_mask)

# input_ids形状：[batch_size, sequence_length]
input_ids = torch.tensor(
    padded_instructions,
    dtype=torch.long
)

# attention_mask形状：[batch_size, sequence_length]
attention_mask = torch.tensor(
    attention_masks,
    dtype=torch.float32
)

embedding = nn.Embedding(
    num_embeddings=len(vocabulary),
    embedding_dim=4,
    padding_idx=pad_id
)

# [batch_size, sequence_length]
# → [batch_size, sequence_length, embedding_dim]
embedded_tokens = embedding(input_ids)

# 增加最后一个维度，便于和token向量逐位置相乘：
# [batch_size, sequence_length]
# → [batch_size, sequence_length, 1]
expanded_mask = attention_mask.unsqueeze(dim=-1)

# 广播（broadcasting）：
# 真实token向量乘1，保持不变；
# padding向量乘0，变成全0。
masked_embeddings = (
    embedded_tokens
    * expanded_mask
)

# 将每条指令中所有真实token向量相加：
# [batch_size, sequence_length, embedding_dim]
# → [batch_size, embedding_dim]
summed_embeddings = masked_embeddings.sum(dim=1)

# 统计每条指令有多少个真实token：
# [batch_size, sequence_length, 1]
# → [batch_size, 1]
valid_token_counts = expanded_mask.sum(dim=1)

# 掩码平均池化（masked mean pooling）：
# 每条指令的token向量总和 ÷ 真实token数量
sentence_features = (
    summed_embeddings
    / valid_token_counts
)

# 使用句子特征产生三个候选动作的logits
classifier = nn.Linear(
    in_features=4,
    out_features=3
)
logits = classifier(sentence_features)
predicted_action_ids = logits.argmax(dim=1)

action_names = [
    "move forward",
    "turn left",
    "turn right"
]

print("编码后但尚未padding：")
for instruction, token_ids in zip(
    instructions,
    encoded_instructions
):
    print(f"{instruction:<30} → {token_ids}")

print("\n补齐后的input_ids：")
print(input_ids)
print("\nattention_mask：")
print(attention_mask)

print("\n形状变化：")
print("input_ids：", input_ids.shape)
print("attention_mask：", attention_mask.shape)
print("embedded_tokens：", embedded_tokens.shape)
print("expanded_mask：", expanded_mask.shape)
print("summed_embeddings：", summed_embeddings.shape)
print("valid_token_counts：", valid_token_counts.shape)
print("sentence_features：", sentence_features.shape)
print("logits：", logits.shape)

print("\n当前随机预测：")
for index, instruction in enumerate(instructions):
    action_id = predicted_action_ids[index].item()
    print(
        f"{instruction:<30} → "
        f"{action_names[action_id]}"
    )

print("\n注意：Embedding和分类头尚未训练，动作预测没有实际意义。")
