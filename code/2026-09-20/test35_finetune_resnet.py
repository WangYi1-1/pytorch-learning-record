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
# 微调（fine-tuning）：在预训练参数的基础上继续训练部分或全部网络。
# 解冻（unfreeze）：把requires_grad从False改回True。
# 参数组（parameter group）：让不同部分的参数使用不同的优化器设置。
# 差分学习率（differential learning rates）：
# 给预训练层和新分类头设置不同的学习率。
# 灾难性遗忘（catastrophic forgetting）：
# 学习率过大时，预训练模型可能快速破坏原来学到的能力。


def evaluate(model, dataloader, loss_fn, device):
    model.eval()

    loss_sum = 0.0
    correct_count = 0
    total_count = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            # 前向传播
            logits = model(images)

            # 损失计算
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

feature_extractor_path = (
    model_folder
    / "best_resnet18_transfer.pth"
)
finetuned_model_path = (
    model_folder
    / "best_resnet18_finetuned.pth"
)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# 使用与test34相同的ResNet18图片预处理
weights = ResNet18_Weights.DEFAULT
transform = weights.transforms()

source_dataset = datasets.CIFAR10(
    root=data_folder,
    train=True,
    download=False,
    transform=transform
)

# 使用与test34完全相同的数据划分
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

# 先创建与test34相同的模型结构。
# weights=None表示这里不重新加载ImageNet权重，
# 因为下一步会加载test34保存的完整state_dict。
model = resnet18(weights=None)
feature_count = model.fc.in_features
model.fc = nn.Linear(feature_count, 10)

model.load_state_dict(
    torch.load(
        feature_extractor_path,
        map_location=device,
        weights_only=True
    )
)
model = model.to(device)

# 第一步：再次冻结整个模型
for parameter in model.parameters():
    parameter.requires_grad = False

# 第二步：只解冻最后一组残差层layer4
for parameter in model.layer4.parameters():
    parameter.requires_grad = True

# 第三步：解冻新的分类头fc
for parameter in model.fc.parameters():
    parameter.requires_grad = True

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

# 参数组：
# layer4保留了有用的预训练参数，所以使用较小学习率；
# fc是针对CIFAR-10的新分类头，可以使用相对较大的学习率。
optimizer = torch.optim.Adam([
    {
        "params": model.layer4.parameters(),
        "lr": 0.00001
    },
    {
        "params": model.fc.parameters(),
        "lr": 0.0001
    }
])

# 先测量微调之前，也就是test34最佳模型的验证结果
initial_loss, initial_accuracy = evaluate(
    model,
    validation_dataloader,
    loss_fn,
    device
)

best_validation_accuracy = initial_accuracy
torch.save(model.state_dict(), finetuned_model_path)

epochs = 3

print("使用设备：", device)
print("模型总参数量：", total_parameter_count)
print("解冻后的可训练参数量：", trainable_parameter_count)
print(
    "微调前验证准确率：",
    f"{initial_accuracy * 100:.2f}%"
)

for epoch in range(1, epochs + 1):
    # 先把整个模型切换到训练模式
    model.train()

    # 当前数据量较小，因此固定所有BatchNorm的统计数据，
    # 防止预训练阶段学到的均值和方差发生剧烈变化。
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()

    train_loss_sum = 0.0
    train_correct = 0
    train_total = 0

    for images, labels in train_dataloader:
        images = images.to(device)
        labels = labels.to(device)

        # 前向传播
        logits = model(images)

        # 损失计算
        loss = loss_fn(logits, labels)

        # 梯度清零
        optimizer.zero_grad()

        # 反向传播：
        # 只为layer4和fc中requires_grad=True的参数计算梯度
        loss.backward()

        # 参数更新：
        # layer4和fc按照各自参数组的学习率更新
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
        torch.save(model.state_dict(), finetuned_model_path)
        print("  已保存当前最佳微调模型")

improvement = (
    best_validation_accuracy
    - initial_accuracy
)

print("\n微调完成")
print(
    "微调前验证准确率：",
    f"{initial_accuracy * 100:.2f}%"
)
print(
    "微调后最佳验证准确率：",
    f"{best_validation_accuracy * 100:.2f}%"
)
print(
    "准确率变化：",
    f"{improvement * 100:+.2f}个百分点"
)
print("最佳微调模型保存在：", finetuned_model_path)
