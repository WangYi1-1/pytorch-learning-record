import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


sys.stdout.reconfigure(encoding="utf-8")
torch.manual_seed(42)


action_tokens = [
    "forward",
    "left",
    "right",
    "stop"
]

action_names = [
    "move forward",
    "turn left",
    "turn right",
    "stop"
]


def make_samples(second_actions, templates):
    """
    根据模板批量生成指令。

    标签始终是第一个动作的编号，
    模型需要学会“先执行第一个动作”这条规则。
    """
    generated_samples = []

    for first_label, first_action in enumerate(action_tokens):
        for second_action in second_actions:
            for template in templates:
                instruction = template.format(
                    first=first_action,
                    second=second_action
                )

                generated_samples.append((
                    instruction,
                    first_label
                ))

    return generated_samples


# 训练集：第二个动作只使用left/right，并使用三种句式。
training_samples = make_samples(
    second_actions=["left", "right"],
    templates=[
        "{first} then {second}",
        "{first} and then {second}",
        "{first} next {second}"
    ]
)

# 验证集：第二个动作换成forward，before没在训练集出现。
validation_samples = make_samples(
    second_actions=["forward"],
    templates=[
        "{first} before {second}"
    ]
)

# 测试集：第二个动作换成stop，followed/by也没在训练集出现。
test_samples = make_samples(
    second_actions=["stop"],
    templates=[
        "{first} followed by {second}"
    ]
)


def build_vocabulary(samples):
    """只使用训练集构建词表，避免提前看到验证/测试信息。"""
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
    expanded_mask = attention_mask.unsqueeze(-1)

    return (
        (token_features * expanded_mask).sum(dim=1)
        / expanded_mask.sum(dim=1)
    )


class TransformerClassifier(nn.Module):

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
            dropout=0.1,
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
        ).unsqueeze(0).expand(
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


def evaluate(model, dataloader, loss_fn, device):
    """在不记录梯度的情况下计算平均损失和准确率。"""
    model.eval()

    loss_sum = 0.0
    correct_count = 0
    total_count = 0

    with torch.no_grad():
        for input_ids, attention_mask, labels in dataloader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            logits = model(input_ids, attention_mask)
            loss = loss_fn(logits, labels)

            batch_size = len(labels)
            loss_sum += loss.item() * batch_size
            correct_count += (
                logits.argmax(dim=1) == labels
            ).sum().item()
            total_count += batch_size

    return (
        loss_sum / total_count,
        correct_count / total_count
    )


def print_predictions(model, dataset, dataloader, device):
    """打印测试集中每条指令的预测结果。"""
    model.eval()
    prediction_index = 0

    with torch.no_grad():
        for input_ids, attention_mask, labels in dataloader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            logits = model(input_ids, attention_mask)
            predictions = logits.argmax(dim=1).cpu()

            for batch_index, predicted_label in enumerate(
                predictions.tolist()
            ):
                instruction, correct_label = dataset.samples[
                    prediction_index
                ]

                print(
                    f"{instruction:<28} | "
                    f"正确={action_names[correct_label]:<12} | "
                    f"预测={action_names[predicted_label]}"
                )

                prediction_index += 1


vocabulary = build_vocabulary(training_samples)
pad_id = vocabulary["<pad>"]

train_dataset = InstructionDataset(
    training_samples,
    vocabulary
)
validation_dataset = InstructionDataset(
    validation_samples,
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
validation_dataloader = DataLoader(
    validation_dataset,
    batch_size=4,
    shuffle=False,
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

model = TransformerClassifier(
    vocabulary_size=len(vocabulary),
    embedding_dim=16,
    max_length=8,
    action_count=len(action_names),
    pad_id=pad_id
).to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.005
)

project_folder = Path(__file__).resolve().parent.parent
model_folder = project_folder / "models"
model_folder.mkdir(exist_ok=True)
model_path = model_folder / "best_transformer_instruction.pth"

epochs = 100
best_validation_accuracy = 0.0
best_validation_loss = float("inf")

print("使用设备：", device)
print("词表：", vocabulary)
print("训练集样本数：", len(train_dataset))
print("验证集样本数：", len(validation_dataset))
print("测试集样本数：", len(test_dataset))

for epoch in range(1, epochs + 1):
    model.train()

    train_loss_sum = 0.0
    train_correct = 0
    train_total = 0

    for input_ids, attention_mask, labels in train_dataloader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        logits = model(input_ids, attention_mask)
        loss = loss_fn(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = len(labels)
        train_loss_sum += loss.item() * batch_size
        train_correct += (
            logits.argmax(dim=1) == labels
        ).sum().item()
        train_total += batch_size

    train_loss = train_loss_sum / train_total
    train_accuracy = train_correct / train_total

    validation_loss, validation_accuracy = evaluate(
        model,
        validation_dataloader,
        loss_fn,
        device
    )

    # 先比验证准确率；准确率相同时，选择验证损失更低的模型。
    is_better_model = (
        validation_accuracy > best_validation_accuracy
        or (
            validation_accuracy == best_validation_accuracy
            and validation_loss < best_validation_loss
        )
    )

    if is_better_model:
        best_validation_accuracy = validation_accuracy
        best_validation_loss = validation_loss

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "vocabulary": vocabulary,
                "action_names": action_names,
                "embedding_dim": 16,
                "max_length": 8
            },
            model_path
        )

    if epoch == 1 or epoch % 20 == 0:
        print(
            f"第{epoch:3d}轮 | "
            f"训练损失={train_loss:.4f} | "
            f"训练准确率={train_accuracy * 100:6.2f}% | "
            f"验证损失={validation_loss:.4f} | "
            f"验证准确率={validation_accuracy * 100:6.2f}%"
        )


# 新建同结构模型，读取验证集表现最好时的参数。
checkpoint = torch.load(
    model_path,
    map_location=device,
    weights_only=True
)

best_model = TransformerClassifier(
    vocabulary_size=len(checkpoint["vocabulary"]),
    embedding_dim=checkpoint["embedding_dim"],
    max_length=checkpoint["max_length"],
    action_count=len(checkpoint["action_names"]),
    pad_id=checkpoint["vocabulary"]["<pad>"]
).to(device)

best_model.load_state_dict(
    checkpoint["model_state_dict"]
)

test_loss, test_accuracy = evaluate(
    best_model,
    test_dataloader,
    loss_fn,
    device
)

print("\n加载最佳模型后的测试结果：")
print(f"测试损失：{test_loss:.4f}")
print(f"测试准确率：{test_accuracy * 100:.2f}%")
print_predictions(
    best_model,
    test_dataset,
    test_dataloader,
    device
)

print("\n最佳模型已保存到：", model_path)
