from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


class CifarCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.features(x)
        logits = self.classifier(x)
        return logits


def evaluate(model, dataloader, loss_fn, device):
    model.eval()

    loss_sum = 0.0
    correct_count = 0
    total_count = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = loss_fn(logits, labels)

            batch_size = len(labels)
            loss_sum += loss.item() * batch_size
            correct_count += (
                logits.argmax(dim=1) == labels
            ).sum().item()
            total_count += batch_size

    average_loss = loss_sum / total_count
    accuracy = correct_count / total_count

    return average_loss, accuracy


train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5)
    )
])

evaluation_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5)
    )
])

train_source = datasets.CIFAR10(
    root=r"D:\codexwork\pytorch学习\data",
    train=True,
    download=False,
    transform=train_transform
)

validation_source = datasets.CIFAR10(
    root=r"D:\codexwork\pytorch学习\data",
    train=True,
    download=False,
    transform=evaluation_transform
)

test_dataset = datasets.CIFAR10(
    root=r"D:\codexwork\pytorch学习\data",
    train=False,
    download=False,
    transform=evaluation_transform
)

generator = torch.Generator().manual_seed(42)
random_indices = torch.randperm(
    len(train_source),
    generator=generator
)

train_indices = random_indices[:45000]
validation_indices = random_indices[45000:]

train_dataset = Subset(train_source, train_indices)
validation_dataset = Subset(
    validation_source,
    validation_indices
)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

validation_dataloader = DataLoader(
    validation_dataset,
    batch_size=64,
    shuffle=False
)

test_dataloader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = CifarCNN().to(device)
loss_fn = nn.CrossEntropyLoss()

# Adam会为不同参数自动调整更新幅度
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

model_folder = Path(__file__).resolve().parent.parent / "models"
model_folder.mkdir(exist_ok=True)
best_model_path = model_folder / "best_cifar_cnn.pth"

best_validation_accuracy = 0.0
epochs = 5

print("使用的设备：", device)


for epoch in range(1, epochs + 1):
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

    validation_loss, validation_accuracy = evaluate(
        model,
        validation_dataloader,
        loss_fn,
        device
    )

    print(
        f"第{epoch}轮 | "
        f"训练损失={train_loss:.4f}，"
        f"训练准确率={train_accuracy * 100:.2f}% | "
        f"验证损失={validation_loss:.4f}，"
        f"验证准确率={validation_accuracy * 100:.2f}%"
    )

    # 只保留验证集准确率最高的模型参数
    if validation_accuracy > best_validation_accuracy:
        best_validation_accuracy = validation_accuracy
        torch.save(model.state_dict(), best_model_path)
        print("  已保存当前最佳模型")


# 加载验证集表现最好的参数，而不是直接使用最后一轮参数
model.load_state_dict(
    torch.load(
        best_model_path,
        map_location=device,
        weights_only=True
    )
)

test_loss, test_accuracy = evaluate(
    model,
    test_dataloader,
    loss_fn,
    device
)

print("训练结束")
print("最佳验证准确率：", f"{best_validation_accuracy * 100:.2f}%")
print("最终测试损失：", f"{test_loss:.4f}")
print("最终测试准确率：", f"{test_accuracy * 100:.2f}%")
print("最佳模型参数：", best_model_path)


