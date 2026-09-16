import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path


# 训练集：用来计算梯度、更新模型参数
train_dataset = datasets.FashionMNIST(
    root=r"D:\codexwork\pytorch学习\data",
    train=True,
    download=True,
    transform=transforms.ToTensor()
)

# 测试集：只用来检查模型，不参与参数更新
test_dataset = datasets.FashionMNIST(
    root=r"D:\codexwork\pytorch学习\data",
    train=False,
    download=True,
    transform=transforms.ToTensor()
)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

test_dataloader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)


class ImageMLP(nn.Module):

    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layer1 = nn.Linear(784, 128)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.layer1(x)
        x = self.relu(x)
        logits = self.layer2(x)
        return logits

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = ImageMLP().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# 训练3轮
for epoch in range(1, 4):
    model.train()

    for batch_images, batch_labels in train_dataloader:
        batch_images = batch_images.to(device)
        batch_labels = batch_labels.to(device)

        logits = model(batch_images)
        loss = loss_fn(logits, batch_labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 每训练完一轮，就用测试集检查一次
    model.eval()

    test_loss_sum = 0.0
    correct_count = 0
    total_count = 0

    # 测试时只做前向计算，不需要建立求导用的计算图
    with torch.no_grad():
        for batch_images, batch_labels in test_dataloader:
            batch_images = batch_images.to(device)
            batch_labels = batch_labels.to(device)

            logits = model(batch_images)
            loss = loss_fn(logits, batch_labels)

            predicted_classes = logits.argmax(dim=1)

            test_loss_sum += loss.item()
            correct_count += (
                predicted_classes == batch_labels
            ).sum().item()
            total_count += len(batch_labels)

    average_test_loss = test_loss_sum / len(test_dataloader)
    test_accuracy = correct_count / total_count

    print(
        f"第{epoch}轮，"
        f"测试损失={average_test_loss:.4f}，"
        f"测试准确率={test_accuracy * 100:.2f}%"
    )


# 创建保存模型参数的文件夹
model_folder = Path(__file__).resolve().parent.parent / "models"
model_folder.mkdir(exist_ok=True)

# 只保存模型训练得到的参数
model_path = model_folder / "fashion_mlp.pth"
torch.save(model.state_dict(), model_path)

print("模型参数已保存到：", model_path)

