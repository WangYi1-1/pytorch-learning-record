import torch
from torch import nn

# 本课常用术语：
# 文本指令（instruction）：模型需要理解的自然语言命令。
# 分词（tokenization）：把一整句话拆成多个token。
# token：模型处理文字时使用的基本单位，这里暂时把一个英文单词当作一个token。
# 词表（vocabulary）：token与整数编号之间的对应表。
# token ID：token在词表中的整数编号。
# 词嵌入（embedding）：将离散的token ID转换成可学习的连续向量。
# 句子表示（sentence representation）：用一个向量表示整条指令。
# logits：模型为每个候选类别输出的原始分数。


torch.manual_seed(42)

# 导航指令
instruction = "turn left and move forward"

# 词表：
# <pad>以后用于补齐不同长度的句子；
# <unk>表示词表中没有出现过的未知单词。
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

# 分词：当前先使用最简单的空格切分
tokens = instruction.lower().split()

# 数值化（numericalization）：
# 将每个token转换成对应的token ID；
# 如果单词不在词表中，就使用<unk>的编号1。
token_ids = [
    vocabulary.get(
        token,
        vocabulary["<unk>"]
    )
    for token in tokens
]

# 神经网络不能直接计算Python字符串，
# 所以把token ID转换成长整型Tensor。
# 最外层中括号表示增加batch维度，目前batch_size=1。
input_ids = torch.tensor(
    [token_ids],
    dtype=torch.long
)

# 词嵌入层（embedding layer）：
# num_embeddings表示词表中一共有多少个token；
# embedding_dim表示每个token用多少个数字表示。
embedding = nn.Embedding(
    num_embeddings=len(vocabulary),
    embedding_dim=4,
    padding_idx=vocabulary["<pad>"]
)

# 嵌入查找（embedding lookup）：
# 根据每个token ID，从embedding.weight中取出对应的一行向量。
embedded_tokens = embedding(input_ids)

# 平均池化（mean pooling）：
# 对一句话中所有token向量求平均，
# 将多个token向量合成一个句子向量。
sentence_features = embedded_tokens.mean(dim=1)

# 动作分类头：
# 输入4维句子特征，输出3个候选动作的原始分数。
classifier = nn.Linear(
    in_features=4,
    out_features=3
)

logits = classifier(sentence_features)
predicted_action_id = logits.argmax(dim=1).item()

action_names = [
    "move forward",
    "turn left",
    "turn right"
]

print("原始指令：", instruction)
print("分词结果：", tokens)
print("token编号：", token_ids)
print("input_ids形状：", input_ids.shape)
print("embedding权重形状：", embedding.weight.shape)
print("所有token向量形状：", embedded_tokens.shape)
print("句子特征形状：", sentence_features.shape)
print("动作logits形状：", logits.shape)
print("动作logits：", logits)
print("当前预测动作：", action_names[predicted_action_id])
print("\n注意：embedding和分类头还没有训练，当前动作预测没有实际意义。")
