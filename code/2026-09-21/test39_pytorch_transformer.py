import sys

import torch
from torch import nn


sys.stdout.reconfigure(encoding="utf-8")


# 本课学习目标：
# 1. 把以前用NumPy手写的Transformer知识，迁移到PyTorch官方层。
# 2. 看清词向量、位置向量、Transformer Encoder的张量形状。
# 3. 验证普通平均池化无法区分词序，Transformer加位置信息后可以区分。


torch.manual_seed(42)

# 微型词表：
# <pad>=0, turn=1, left=2, then=3, right=4
vocabulary_size = 5
pad_id = 0

# 两句话包含完全相同的token，只是left和right顺序相反。
input_ids = torch.tensor([
    [1, 2, 3, 4],  # turn left then right
    [1, 4, 3, 2]   # turn right then left
])

# 当前没有padding，所以所有位置都是真token，全部标记为1。
attention_mask = torch.ones_like(
    input_ids,
    dtype=torch.float32
)


class TransformerTextClassifier(nn.Module):

    def __init__(
        self,
        vocabulary_size,
        embedding_dim,
        max_length,
        action_count,
        pad_id
    ):
        super().__init__()

        # Token Embedding：每个token ID查表得到一个词向量。
        self.token_embedding = nn.Embedding(
            num_embeddings=vocabulary_size,
            embedding_dim=embedding_dim,
            padding_idx=pad_id
        )

        # Positional Embedding：给第0、第1、第2……个位置分配可训练向量。
        self.position_embedding = nn.Embedding(
            num_embeddings=max_length,
            embedding_dim=embedding_dim
        )

        # Transformer Encoder Layer：
        # 一层 = 多头自注意力 + 残差连接 + LayerNorm + FFN。
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=4,
            dim_feedforward=32,
            dropout=0.0,
            batch_first=True
        )

        # Transformer Encoder：将上面的Encoder Layer叠加指定次数。
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=1,
            enable_nested_tensor=False
        )

        # 分类头：把整句话的特征变成各个动作的logits。
        self.classifier = nn.Linear(
            embedding_dim,
            action_count
        )

    def forward(self, input_ids, attention_mask):
        batch_size, sequence_length = input_ids.shape

        # [batch, sequence]
        # → [batch, sequence, embedding_dim]
        token_features = self.token_embedding(input_ids)

        # 生成位置编号：[0, 1, 2, ..., sequence_length-1]
        position_ids = torch.arange(
            sequence_length,
            device=input_ids.device
        )

        # [sequence]扩展为[batch, sequence]，每句话共用同一套位置编号。
        position_ids = position_ids.unsqueeze(0).expand(
            batch_size,
            -1
        )

        # [batch, sequence]
        # → [batch, sequence, embedding_dim]
        position_features = self.position_embedding(
            position_ids
        )

        # 词向量+位置向量，让模型同时知道“是什么词”和“在哪里”。
        features = token_features + position_features

        # PyTorch的src_key_padding_mask与我们的attention_mask含义相反：
        # attention_mask：1=真token，0=padding
        # padding_mask：False=真token，True=padding
        padding_mask = attention_mask == 0

        # 编码后形状不变：
        # [batch, sequence, embedding_dim]
        # → [batch, sequence, embedding_dim]
        encoded_tokens = self.encoder(
            features,
            src_key_padding_mask=padding_mask
        )

        # 掩码平均池化：将多个token特征汇总为一个句子特征。
        expanded_mask = attention_mask.unsqueeze(-1)
        sentence_features = (
            (encoded_tokens * expanded_mask).sum(dim=1)
            / expanded_mask.sum(dim=1)
        )

        # [batch, embedding_dim]
        # → [batch, action_count]
        logits = self.classifier(sentence_features)
        return logits, token_features, encoded_tokens, sentence_features


model = TransformerTextClassifier(
    vocabulary_size=vocabulary_size,
    embedding_dim=16,
    max_length=10,
    action_count=4,
    pad_id=pad_id
)

logits, token_features, encoded_tokens, sentence_features = model(
    input_ids,
    attention_mask
)

# 先看普通平均池化：两句话token相同，只是顺序不同，结果完全相同。
mean_pooled_features = token_features.mean(dim=1)

print("input_ids形状：", input_ids.shape)
print("token_features形状：", token_features.shape)
print("encoded_tokens形状：", encoded_tokens.shape)
print("sentence_features形状：", sentence_features.shape)
print("logits形状：", logits.shape)

print("\n普通平均池化后的两个向量是否相同：")
print(torch.allclose(
    mean_pooled_features[0],
    mean_pooled_features[1]
))

print("\n加入位置信息并经过Transformer后，两个句子向量是否相同：")
print(torch.allclose(
    sentence_features[0],
    sentence_features[1]
))

print("\n未训练模型输出的logits（只用于验证前向传播，还不是有意义的预测）：")
print(logits)

