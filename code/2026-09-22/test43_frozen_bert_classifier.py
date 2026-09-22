import os
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer
from transformers.utils import logging


sys.stdout.reconfigure(encoding="utf-8")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
logging.set_verbosity_error()
torch.manual_seed(42)


model_name = "google/bert_uncased_L-2_H-128_A-2"

action_names = [
    "move forward",
    "turn left",
    "turn right",
    "stop"
]

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

validation_samples = [
    ("head straight", 0),
    ("take a left", 1),
    ("make a right", 2),
    ("remain here", 3)
]

test_samples = [
    ("proceed forward", 0),
    ("continue ahead", 0),
    ("take the left path", 1),
    ("rotate to the left", 1),
    ("take the right path", 2),
    ("rotate to the right", 2),
    ("wait at this place", 3),
    ("do not go anywhere", 3)
]


class InstructionDataset(Dataset):
    """Dataset只保存原始文字和标签，不再自己构建词表。"""

    def __init__(self, samples):
        super().__init__()
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        instruction, label = self.samples[index]
        return instruction, label


# 从本地缓存加载test42已经下载的Tokenizer。
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    local_files_only=True
)


def collate_batch(batch):
    """Tokenizer直接将一批原始文字整理成BERT输入。"""
    instructions, labels = zip(*batch)

    encoded_inputs = tokenizer(
        list(instructions),
        padding=True,
        truncation=True,
        max_length=32,
        return_tensors="pt"
    )

    labels = torch.tensor(
        labels,
        dtype=torch.long
    )

    return encoded_inputs, labels


class BertActionClassifier(nn.Module):

    def __init__(self, model_name, action_count):
        super().__init__()

        # 加载已经预训练好的BERT文本骨干网络。
        self.text_encoder = AutoModel.from_pretrained(
            model_name,
            local_files_only=True
        )

        # 冻结骨干网络：不计算BERT参数的梯度，也不更新它们。
        for parameter in self.text_encoder.parameters():
            parameter.requires_grad = False

        hidden_size = self.text_encoder.config.hidden_size

        # 新的动作分类头：128维BERT特征 → 4个动作logits。
        self.classifier = nn.Linear(
            hidden_size,
            action_count
        )

    def forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids=None
    ):
        outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        # [batch, sequence, 128] → [batch, 128]
        cls_features = outputs.last_hidden_state[:, 0, :]

        # [batch, 128] → [batch, 4]
        return self.classifier(cls_features)


def evaluate(model, dataloader, loss_fn, device):
    model.eval()

    loss_sum = 0.0
    correct_count = 0
    total_count = 0

    with torch.no_grad():
        for encoded_inputs, labels in dataloader:
            encoded_inputs = encoded_inputs.to(device)
            labels = labels.to(device)

            logits = model(**encoded_inputs)
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
    model.eval()
    prediction_index = 0

    with torch.no_grad():
        for encoded_inputs, labels in dataloader:
            del labels

            encoded_inputs = encoded_inputs.to(device)
            logits = model(**encoded_inputs)
            predictions = logits.argmax(dim=1).cpu()

            for predicted_label in predictions.tolist():
                instruction, correct_label = dataset.samples[
                    prediction_index
                ]

                print(
                    f"{instruction:<25} | "
                    f"正确={action_names[correct_label]:<12} | "
                    f"预测={action_names[predicted_label]}"
                )

                prediction_index += 1


train_dataset = InstructionDataset(training_samples)
validation_dataset = InstructionDataset(validation_samples)
test_dataset = InstructionDataset(test_samples)

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

model = BertActionClassifier(
    model_name=model_name,
    action_count=len(action_names)
).to(device)

total_parameter_count = sum(
    parameter.numel()
    for parameter in model.parameters()
)
trainable_parameter_count = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)

loss_fn = nn.CrossEntropyLoss()

# 优化器只接收新分类头的参数。
optimizer = torch.optim.Adam(
    model.classifier.parameters(),
    lr=0.01
)

project_folder = Path(__file__).resolve().parent.parent
model_folder = project_folder / "models"
model_folder.mkdir(exist_ok=True)
model_path = model_folder / "best_frozen_bert_action.pth"

epochs = 50
best_validation_accuracy = 0.0
best_validation_loss = float("inf")

print("使用设备：", device)
print("模型总参数：", total_parameter_count)
print("可训练参数：", trainable_parameter_count)
print("冻结参数：", total_parameter_count - trainable_parameter_count)

for epoch in range(1, epochs + 1):
    model.train()

    # model.train()会递归让所有子模块进入训练模式。
    # BERT已经冻结，所以再单独设为eval，关闭它内部的Dropout。
    model.text_encoder.eval()

    train_loss_sum = 0.0
    train_correct = 0
    train_total = 0

    for encoded_inputs, labels in train_dataloader:
        encoded_inputs = encoded_inputs.to(device)
        labels = labels.to(device)

        logits = model(**encoded_inputs)
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

        # BERT参数来自预训练模型且未被修改，这里只保存新分类头。
        torch.save(
            {
                "classifier_state_dict": (
                    model.classifier.state_dict()
                ),
                "model_name": model_name,
                "action_names": action_names
            },
            model_path
        )

    if epoch == 1 or epoch % 10 == 0:
        print(
            f"第{epoch:2d}轮 | "
            f"训练损失={train_loss:.4f} | "
            f"训练准确率={train_accuracy * 100:6.2f}% | "
            f"验证准确率={validation_accuracy * 100:6.2f}%"
        )


checkpoint = torch.load(
    model_path,
    map_location=device,
    weights_only=True
)

best_model = BertActionClassifier(
    model_name=checkpoint["model_name"],
    action_count=len(checkpoint["action_names"])
).to(device)

best_model.classifier.load_state_dict(
    checkpoint["classifier_state_dict"]
)

test_loss, test_accuracy = evaluate(
    best_model,
    test_dataloader,
    loss_fn,
    device
)

print("\n冻结BERT骨干网络的测试结果：")
print(f"测试损失：{test_loss:.4f}")
print(f"测试准确率：{test_accuracy * 100:.2f}%")
print_predictions(
    best_model,
    test_dataset,
    test_dataloader,
    device
)

print("\n最佳分类头已保存到：", model_path)
