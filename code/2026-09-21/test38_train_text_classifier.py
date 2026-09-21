from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

# 本课常用术语：
# 文本分类（text classification）：根据文字内容预测一个类别。
# 自定义Dataset：定义一条数据应当怎样读取和转换。
# collate_fn：定义DataLoader怎样将多条数据组成一个batch。
# 动态padding：只补到当前batch中的最长长度。
# 句子编码器（sentence encoder）：将整条指令转换成特征向量。
# 泛化（generalization）：模型处理训练时没有见过的新样本的能力。


torch.manual_seed(42)

action_names = [
    "move forward",
    "turn left",
    "turn right",
    "stop"
]

# 每条数据由“文字指令”和“正确动作编号”组成
training_samples = [
    ("go forward", 0),
    ("move forward", 0),
    ("walk forward", 0),
    ("continue forward", 0),
    ("go straight", 0),
    ("move straight", 0),
    ("walk straight ahead", 0),
    ("continue straight", 0),

    ("turn left", 1),
    ("go left", 1),
    ("move left", 1),
    ("walk left", 1),
    ("rotate left", 1),
    ("head left", 1),
    ("take the left direction", 1),
    ("make a left turn", 1),

    ("turn right", 2),
    ("go right", 2),
    ("move right", 2),
    ("walk right", 2),
    ("rotate right", 2),
    ("head right", 2),
    ("take the right direction", 2),
    ("make a right turn", 2),

    ("stop", 3),
    ("halt", 3),
    ("wait here", 3),
    ("stay here", 3),
    ("remain still", 3),
    ("do not move", 3),
    ("stop moving", 3),
    ("finish here", 3)
]

# 测试样本没有原样出现在训练集中
test_samples = [
    ("head straight", 0),
    ("take a left", 1),
    ("make a right", 2),
    ("remain here", 3)
]


def build_vocabulary(samples):
    # 先保留padding和未知词两个特殊token
    vocabulary = {
        "<pad>": 0,
        "<unk>": 1
    }

    for instruction, label in samples:
        # 当前只使用文字，label暂时不参与词表构建
        del label

        tokens = instruction.lower().split()

        for token in tokens:
            if token not in vocabulary:
                # 新token的编号等于当前词表长度
                vocabulary[token] = len(vocabulary)

    return vocabulary


class InstructionDataset(Dataset):

    def __init__(self, samples, vocabulary):
        super().__init__()
        self.samples = samples
        self.vocabulary = vocabulary
        self.unknown_id = vocabulary["<unk>"]

    def __len__(self):
        # Dataset中一共有多少条数据
        return len(self.samples)

    def __getitem__(self, index):
        # 取出第index条文字和正确动作编号
        instruction, label = self.samples[index]

        tokens = instruction.lower().split()
        token_ids = [
            self.vocabulary.get(
                token,
                self.unknown_id
            )
            for token in tokens
        ]

        # 每条指令暂时保持自己的真实长度，
        # padding将在DataLoader组成batch时完成。
        return (
            torch.tensor(token_ids, dtype=torch.long),
            torch.tensor(label, dtype=torch.long)
        )


def collate_batch(batch):
    # batch是Dataset返回的多条(token_ids, label)
    token_sequences, labels = zip(*batch)

    batch_size = len(token_sequences)
    max_length = max(
        len(token_ids)
        for token_ids in token_sequences
    )

    # 创建已经填满pad_id的input_ids
    input_ids = torch.full(
        size=(batch_size, max_length),
        fill_value=pad_id,
        dtype=torch.long
    )

    # attention_mask开始全部为0
    attention_mask = torch.zeros(
        size=(batch_size, max_length),
        dtype=torch.float32
    )

    for row_index, token_ids in enumerate(
        token_sequences
    ):
        sequence_length = len(token_ids)

        # 将真实token ID写入当前行前面的位置
        input_ids[
            row_index,
            :sequence_length
        ] = token_ids

        # 真实token位置标记为1，其余padding位置保持为0
        attention_mask[
            row_index,
            :sequence_length
        ] = 1

    # labels原来是多个单独Tensor，stack后组成[batch_size]
    labels = torch.stack(labels)
    return input_ids, attention_mask, labels


class TextActionClassifier(nn.Module):

    def __init__(
        self,
        vocabulary_size,
        embedding_dim,
        action_count,
        pad_id
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocabulary_size,
            embedding_dim=embedding_dim,
            padding_idx=pad_id
        )

        # 分类头：句子特征 → 隐藏特征 → 四个动作logits
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 16),
            nn.ReLU(),
            nn.Linear(16, action_count)
        )

    def forward(self, input_ids, attention_mask):
        # [batch, sequence]
        # → [batch, sequence, embedding_dim]
        embedded_tokens = self.embedding(input_ids)

        # [batch, sequence]
        # → [batch, sequence, 1]
        expanded_mask = attention_mask.unsqueeze(-1)

        # 掩码平均池化，只汇总真实token
        summed_embeddings = (
            embedded_tokens
            * expanded_mask
        ).sum(dim=1)

        valid_token_counts = expanded_mask.sum(dim=1)
        sentence_features = (
            summed_embeddings
            / valid_token_counts
        )

        # [batch, embedding_dim]
        # → [batch, action_count]
        logits = self.classifier(sentence_features)
        return logits


vocabulary = build_vocabulary(training_samples)
pad_id = vocabulary["<pad>"]

train_dataset = InstructionDataset(
    training_samples,
    vocabulary
)
test_dataset = InstructionDataset(
    test_samples,
    vocabulary
)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    collate_fn=collate_batch
)
test_dataloader = DataLoader(
    test_dataset,
    batch_size=4,
    shuffle=False,
    collate_fn=collate_batch
)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = TextActionClassifier(
    vocabulary_size=len(vocabulary),
    embedding_dim=16,
    action_count=len(action_names),
    pad_id=pad_id
).to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.01
)

epochs = 100

print("使用设备：", device)
print("训练样本数：", len(train_dataset))
print("词表大小：", len(vocabulary))

for epoch in range(1, epochs + 1):
    model.train()

    loss_sum = 0.0
    correct_count = 0
    total_count = 0

    for input_ids, attention_mask, labels in train_dataloader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        # 前向传播
        logits = model(input_ids, attention_mask)

        # 损失计算
        loss = loss_fn(logits, labels)

        # 梯度清零
        optimizer.zero_grad()

        # 反向传播：
        # 同时计算Embedding和分类头参数的梯度
        loss.backward()

        # 参数更新
        optimizer.step()

        batch_size = len(labels)
        loss_sum += loss.item() * batch_size
        correct_count += (
            logits.argmax(dim=1) == labels
        ).sum().item()
        total_count += batch_size

    average_loss = loss_sum / total_count
    accuracy = correct_count / total_count

    if epoch == 1 or epoch % 20 == 0:
        print(
            f"第{epoch:3d}轮："
            f"训练损失={average_loss:.4f}，"
            f"训练准确率={accuracy * 100:.2f}%"
        )


# 使用没有原样出现在训练集中的文字检查泛化能力
model.eval()

with torch.no_grad():
    input_ids, attention_mask, labels = next(
        iter(test_dataloader)
    )

    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    logits = model(input_ids, attention_mask)
    predictions = logits.argmax(dim=1).cpu()

print("\n新指令预测：")

for index, (instruction, correct_label) in enumerate(
    test_samples
):
    predicted_label = predictions[index].item()

    print(
        f"{instruction:<20} | "
        f"正确={action_names[correct_label]:<12} | "
        f"预测={action_names[predicted_label]}"
    )

project_folder = Path(__file__).resolve().parent.parent
model_folder = project_folder / "models"
model_folder.mkdir(exist_ok=True)
model_path = model_folder / "text_action_classifier.pth"

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "vocabulary": vocabulary,
        "action_names": action_names
    },
    model_path
)

print("\n模型、词表和动作名称已保存到：", model_path)

