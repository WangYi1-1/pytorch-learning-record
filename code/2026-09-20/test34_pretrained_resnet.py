from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from torchvision.models import (
    ResNet18_Weights,
    resnet18
)

# 本课常用术语：
# 预训练模型（pretrained model）：已经在大型数据集上训练过的模型。
# 迁移学习（transfer learning）：把已有模型学到的能力迁移到新任务。
# 骨干网络（backbone）：负责提取特征的主体网络。
# 分类头（classification head）：根据特征输出新任务的类别分数。
# 冻结参数（freeze parameters）：保留参数数值，不让反向传播更新它们。
# 特征提取（feature extraction）：使用冻结的骨干网络生成图像特征。


def evaluate(model, dataloader, loss_fn, device):
    # 评估模式：BatchNorm等层使用已经保存的统计数据
    model.eval()

    loss_sum = 0.0
    correct_count = 0
    total_count = 0

    # 推理阶段：只进行前向传播，不建立反向传播计算图
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            # 前向传播：ResNet输出10个CIFAR-10类别分数
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


# 本文件在仓库中的位置是code/2026-09-20/，向上三级到仓库根目录
project_folder = Path(__file__).resolve().parents[2]
data_folder = project_folder / "data"
model_folder = project_folder / "models"
model_folder.mkdir(exist_ok=True)

best_model_path = (
    model_folder
    / "best_resnet18_transfer.pth"
)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# DEFAULT表示使用Torchvision推荐的ResNet18预训练权重。
# 第一次运行时会自动下载大约45MB的权重文件。
weights = ResNet18_Weights.DEFAULT

# 使用和预训练阶段相匹配的数据预处理：
# 调整图片大小、转成Tensor，并使用ImageNet的均值和标准差归一化。
transform = ResNet18_Weights.DEFAULT.transforms()

source_dataset = datasets.CIFAR10(
    root=data_folder,
    train=True,
    download=False,
    transform=transform
)

# 固定训练集和验证集划分，保证实验可以复现
split_generator = torch.Generator().manual_seed(42)
random_indices = torch.randperm(
    len(source_dataset),
    generator=split_generator
)

train_indices = random_indices[:10000]
validation_indices = random_indices[10000:12000]

train_dataset = Subset(
    source_dataset,
    train_indices
)
validation_dataset = Subset(
    source_dataset,
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

# 加载模型结构和ImageNet预训练参数
model = resnet18(weights=weights)

# 冻结骨干网络：
# requires_grad=False表示不计算这些参数的梯度，也不更新它们。
for parameter in model.parameters():
    parameter.requires_grad = False

# ResNet18原来的fc层用于1000分类。
# in_features表示送入原分类头的特征数量，ResNet18中是512。
feature_count = model.fc.in_features

# 替换分类头：
# 新Linear层接收512个图像特征，输出10个CIFAR-10类别分数。
# 新创建的fc参数默认requires_grad=True，因此可以训练。
model.fc = nn.Linear(feature_count, 10)
model = model.to(device)

# 参数量（parameter count）：模型一共有多少个可以保存的数字
total_parameter_count = sum(
    parameter.numel()
    for parameter in model.parameters()
)

# 可训练参数量（trainable parameter count）：
# 只统计requires_grad=True的参数
trainable_parameter_count = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)

loss_fn = nn.CrossEntropyLoss()

# 优化器只接收新分类头的参数。
# 骨干网络已经冻结，不参与本次参数更新。
optimizer = torch.optim.Adam(
    model.fc.parameters(),
    lr=0.001
)

epochs = 3
best_validation_accuracy = 0.0

print("使用设备：", device)
print("训练集大小：", len(train_dataset))
print("验证集大小：", len(validation_dataset))
print("ResNet输入特征数：", feature_count)
print("模型总参数量：", total_parameter_count)
print("可训练参数量：", trainable_parameter_count)

for epoch in range(1, epochs + 1):
    # 骨干网络保持评估模式，使冻结的BatchNorm统计数据不发生变化
    model.eval()

    # 分类头进入训练模式。当前fc只有Linear层，
    # 这样写是为了明确“只训练分类头”的意图。
    model.fc.train()

    train_loss_sum = 0.0
    train_correct = 0
    train_total = 0

    for images, labels in train_dataloader:
        images = images.to(device)
        labels = labels.to(device)

        # 前向传播：
        # 冻结骨干提取512个特征，新分类头输出10个类别分数。
        logits = model(images)

        # 损失计算：比较类别分数与正确类别
        loss = loss_fn(logits, labels)

        # 梯度清零
        optimizer.zero_grad()

        # 反向传播：
        # 只有新fc层的参数requires_grad=True，因此只计算它们的梯度。
        loss.backward()

        # 参数更新：只修改新分类头的weight和bias
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

    if validation_accuracy > best_validation_accuracy:
        best_validation_accuracy = validation_accuracy
        torch.save(model.state_dict(), best_model_path)
        print("  已保存当前最佳迁移学习模型")

print("\n迁移学习完成")
print(
    "最佳验证准确率：",
    f"{best_validation_accuracy * 100:.2f}%"
)
print("模型参数保存在：", best_model_path)
