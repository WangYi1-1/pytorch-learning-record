from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class FashionCNN(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.output_layer = nn.Linear(4 * 14 * 14, 10)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.flatten(x)
        logits = self.output_layer(x)
        return logits


train_dataset = datasets.FashionMNIST(
    root=r"D:\codexwork\pytorch学习\data",
    train=True,
    download=True,
    transform=transforms.ToTensor()
)

test_dataset = datasets.FashionMNIST(
    root=r"D:\codexwork\pytorch学习\data",
    train=False,
    download=True,
    transform=transforms.ToTensor()
)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

test_dataloader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = FashionCNN().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

print("使用的设备：", device)


for epoch in range(1, 4):
    # ---------- 训练 ----------
    model.train()

    train_loss_sum = 0.0
    train_correct = 0
    train_total = 0

    for images, labels in train_dataloader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
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

    # ---------- 测试 ----------
    model.eval()

    test_loss_sum = 0.0
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for images, labels in test_dataloader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = loss_fn(logits, labels)

            batch_size = len(labels)
            test_loss_sum += loss.item() * batch_size
            test_correct += (
                logits.argmax(dim=1) == labels
            ).sum().item()
            test_total += batch_size

    test_loss = test_loss_sum / test_total
    test_accuracy = test_correct / test_total

    print(
        f"第{epoch}轮 | "
        f"训练损失={train_loss:.4f}，"
        f"训练准确率={train_accuracy * 100:.2f}% | "
        f"测试损失={test_loss:.4f}，"
        f"测试准确率={test_accuracy * 100:.2f}%"
    )


model_folder = Path(__file__).resolve().parent.parent / "models"
model_folder.mkdir(exist_ok=True)
model_path = model_folder / "fashion_cnn.pth"

torch.save(model.state_dict(), model_path)
print("CNN参数已保存到：", model_path)


