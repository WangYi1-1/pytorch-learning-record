import sys

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


sys.stdout.reconfigure(encoding="utf-8")
torch.manual_seed(42)


# 本课的分类目标：预测指令中“先执行”的动作。
action_names = [
    "move forward",
    "turn left",
    "turn right",
    "stop"
]

# 每两条配对样本包含完全相同的token，只是顺序相反。
# 标签是第一个出现的动作。
samples = [
    ("forward then left", 0),
    ("left then forward", 1),

    ("forward then right", 0),
    ("right then forward", 2),

    ("forward then stop", 0),
    ("stop then forward", 3),

    ("left then right", 1),
    ("right then left", 2),

    ("left then stop", 1),
    ("stop then left", 3),

    ("right then stop", 2),
    ("stop then right", 3)
]


def build_vocabulary(samples):
    vocabulary = {
        "<pad>": 0,
        "<unk>": 1
    }

    for instruction, _ in samples:
        for token in instruction.split():
            if token not in vocabulary:
                vocabulary[token] = len(vocabulary)

    return vocabulary


class InstructionDataset(Dataset):

    def __init__(self, samples, vocabulary):
        super().__init__()
        self.samples = samples
        self.vocabulary = vocabulary
        self.unknown_id = vocabulary["<unk>"]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        instruction, label = self.samples[index]

        token_ids = [
            self.vocabulary.get(token, self.unknown_id)
            for token in instruction.split()
        ]

        return (
            torch.tensor(token_ids, dtype=torch.long),
            torch.tensor(label, dtype=torch.long)
        )


def collate_batch(batch):
    token_sequences, labels = zip(*batch)

    batch_size = len(token_sequences)
    max_length = max(
        len(token_ids)
        for token_ids in token_sequences
    )

    input_ids = torch.full(
        size=(batch_size, max_length),
        fill_value=pad_id,
        dtype=torch.long
    )

    attention_mask = torch.zeros(
        size=(batch_size, max_length),
        dtype=torch.float32
    )

    for row_index, token_ids in enumerate(token_sequences):
        sequence_length = len(token_ids)

        input_ids[
            row_index,
            :sequence_length
        ] = token_ids

        attention_mask[
            row_index,
            :sequence_length
        ] = 1

    labels = torch.stack(labels)
    return input_ids, attention_mask, labels


def masked_mean_pooling(token_features, attention_mask):
    """掩码平均池化：只对真实token特征求平均。"""
    expanded_mask = attention_mask.unsqueeze(-1)

    return (
        (token_features * expanded_mask).sum(dim=1)
        / expanded_mask.sum(dim=1)
    )


class MeanPoolingClassifier(nn.Module):
    """对照组：Embedding后直接平均，不使用位置信息。"""

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

        self.classifier = nn.Linear(
            embedding_dim,
            action_count
        )

    def forward(self, input_ids, attention_mask):
        token_features = self.embedding(input_ids)

        sentence_features = masked_mean_pooling(
            token_features,
            attention_mask
        )

        return self.classifier(sentence_features)


class TransformerClassifier(nn.Module):
    """实验组：Token/Position Embedding + Transformer Encoder。"""

    def __init__(
        self,
        vocabulary_size,
        embedding_dim,
        max_length,
        action_count,
        pad_id
    ):
        super().__init__()

        self.token_embedding = nn.Embedding(
            num_embeddings=vocabulary_size,
            embedding_dim=embedding_dim,
            padding_idx=pad_id
        )

        self.position_embedding = nn.Embedding(
            num_embeddings=max_length,
            embedding_dim=embedding_dim
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=4,
            dim_feedforward=32,
            dropout=0.0,
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=1,
            enable_nested_tensor=False
        )

        self.classifier = nn.Linear(
            embedding_dim,
            action_count
        )

    def forward(self, input_ids, attention_mask):
        batch_size, sequence_length = input_ids.shape

        token_features = self.token_embedding(input_ids)

        position_ids = torch.arange(
            sequence_length,
            device=input_ids.device
        )
        position_ids = position_ids.unsqueeze(0).expand(
            batch_size,
            -1
        )

        position_features = self.position_embedding(
            position_ids
        )

        features = token_features + position_features
        padding_mask = attention_mask == 0

        encoded_tokens = self.encoder(
            features,
            src_key_padding_mask=padding_mask
        )

        sentence_features = masked_mean_pooling(
            encoded_tokens,
            attention_mask
        )

        return self.classifier(sentence_features)


def train_model(model, model_name, dataloader, device):
    """使用相同的损失函数、优化器和训练轮数训练一个模型。"""
    model = model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.01
    )

    epochs = 200

    print(f"\n开始训练：{model_name}")

    for epoch in range(1, epochs + 1):
        model.train()

        loss_sum = 0.0
        correct_count = 0
        total_count = 0

        for input_ids, attention_mask, labels in dataloader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            # 前向传播：计算各动作的logits。
            logits = model(input_ids, attention_mask)

            # 损失计算：比较logits与正确动作编号。
            loss = loss_fn(logits, labels)

            # 梯度清零、反向传播、参数更新。
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size = len(labels)
            loss_sum += loss.item() * batch_size
            correct_count += (
                logits.argmax(dim=1) == labels
            ).sum().item()
            total_count += batch_size

        average_loss = loss_sum / total_count
        accuracy = correct_count / total_count

        if epoch == 1 or epoch % 50 == 0:
            print(
                f"第{epoch:3d}轮："
                f"损失={average_loss:.4f}，"
                f"准确率={accuracy * 100:6.2f}%"
            )

    return model


def evaluate_model(model, model_name, dataloader, device):
    """评估模型，并打印每条指令的预测结果。"""
    model.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for input_ids, attention_mask, labels in dataloader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            logits = model(input_ids, attention_mask)
            predictions = logits.argmax(dim=1).cpu()

            all_predictions.extend(predictions.tolist())
            all_labels.extend(labels.tolist())

    correct_count = sum(
        predicted == correct
        for predicted, correct in zip(
            all_predictions,
            all_labels
        )
    )
    accuracy = correct_count / len(all_labels)

    print(f"\n{model_name}最终结果：")

    for (instruction, correct_label), predicted_label in zip(
        samples,
        all_predictions
    ):
        print(
            f"{instruction:<20} | "
            f"正确={action_names[correct_label]:<12} | "
            f"预测={action_names[predicted_label]}"
        )

    print(f"最终准确率：{accuracy * 100:.2f}%")
    return accuracy


vocabulary = build_vocabulary(samples)
pad_id = vocabulary["<pad>"]

dataset = InstructionDataset(samples, vocabulary)

# 数据集只有12条数据，每轮一次读入全部样本。
dataloader = DataLoader(
    dataset,
    batch_size=len(dataset),
    shuffle=False,
    collate_fn=collate_batch
)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("使用设备：", device)
print("词表：", vocabulary)
print("数据集大小：", len(dataset))

# 对照组：无位置信息、无Transformer。
torch.manual_seed(42)
mean_pooling_model = MeanPoolingClassifier(
    vocabulary_size=len(vocabulary),
    embedding_dim=16,
    action_count=len(action_names),
    pad_id=pad_id
)

mean_pooling_model = train_model(
    mean_pooling_model,
    "普通平均池化模型",
    dataloader,
    device
)

mean_pooling_accuracy = evaluate_model(
    mean_pooling_model,
    "普通平均池化模型",
    dataloader,
    device
)

# 实验组：加入位置信息和Transformer Encoder。
torch.manual_seed(42)
transformer_model = TransformerClassifier(
    vocabulary_size=len(vocabulary),
    embedding_dim=16,
    max_length=10,
    action_count=len(action_names),
    pad_id=pad_id
)

transformer_model = train_model(
    transformer_model,
    "Transformer模型",
    dataloader,
    device
)

transformer_accuracy = evaluate_model(
    transformer_model,
    "Transformer模型",
    dataloader,
    device
)

print("\n对照实验总结：")
print(
    "普通平均池化模型："
    f"{mean_pooling_accuracy * 100:.2f}%"
)
print(
    "Transformer模型："
    f"{transformer_accuracy * 100:.2f}%"
)
print(
    "普通平均池化看不见词序；"
    "Transformer利用位置特征和自注意力区分顺序。"
)
