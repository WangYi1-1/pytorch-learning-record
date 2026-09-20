from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# 本课常用的正式术语：
# 前向传播（forward propagation）：数据经过模型各层，算出预测结果。
# 损失（loss）：用一个数字衡量预测结果和正确答案的差距。
# 反向传播（backpropagation）：从loss出发，计算每个参数的梯度。
# 参数更新（parameter update）：优化器根据梯度修改weight和bias。
# epoch：模型完整看完一次训练集。
# batch：一次送进模型的一小批数据。
# 超参数（hyperparameter）：训练前由人设置的数，例如学习率和batch_size。


class CifarCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            # 第一层卷积（convolution）：
            # 输入3张RGB颜色通道，使用32个卷积核提取局部特征，
            # padding=1使图片长宽保持不变。
            # [batch, 3, 32, 32] → [batch, 32, 32, 32]
            nn.Conv2d(3, 32, kernel_size=3, padding=1),

            # 激活函数（activation function）：
            # 将负数变成0、正数保留，引入非线性，形状不变。
            nn.ReLU(),

            # 最大池化（max pooling）：
            # 每个2×2区域保留最大值，让长宽缩小一半，
            # 通道数量保持不变。
            # [batch, 32, 32, 32] → [batch, 32, 16, 16]
            nn.MaxPool2d(2, 2),

            # 第二层卷积：
            # 接收上一层的32张特征图，使用64个卷积核，
            # 组合较简单的局部特征，形成64张更高级的特征图。
            # [batch, 32, 16, 16] → [batch, 64, 16, 16]
            nn.Conv2d(32, 64, kernel_size=3, padding=1),

            # 第二次非线性激活，形状不变
            nn.ReLU(),

            # 第二次最大池化，再次将长宽缩小一半。
            # [batch, 64, 16, 16] → [batch, 64, 8, 8]
            nn.MaxPool2d(2, 2)
        )

        self.classifier = nn.Sequential(
            # 展平层（Flatten layer）
            nn.Flatten(),

            # 全连接层（Fully connected layer / Linear layer）
            nn.Linear(64 * 8 * 8, 128),

            # 激活函数（Activation function）
            nn.ReLU(),

            # 输出层（Output layer），也是全连接层
            nn.Linear(128, 10)
        )

    def forward(self, x):
        # 前向传播：输入图片经过特征提取和分类器，得到10个类别分数
        x = self.features(x)
        return self.classifier(x)


def evaluate_accuracy(model, dataloader, device):
    # 评估模式：只检查模型效果，不训练参数
    model.eval()

    correct_count = 0
    total_count = 0

    # 推理阶段不需要记录梯度
    with torch.no_grad():
        # 按batch遍历整个验证集
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            # 前向传播：使用当前参数计算类别分数
            logits = model(images)

            # 预测：选择分数最高的类别编号
            predictions = logits.argmax(dim=1)

            # 统计预测正确的样本数量
            correct_count += (
                predictions == labels
            ).sum().item()
            total_count += len(labels)

    return correct_count / total_count


def run_experiment(
    learning_rate,
    train_dataset,
    validation_dataset,
    device,
    epochs=5
):
    # 这叫控制变量实验：
    # 三次实验只改变学习率，其他训练条件尽量保持一致。

    # 随机种子（random seed）：
    # 每次使用相同的随机数起点，使模型初始参数相同，
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # 单独控制DataLoader打乱数据时使用的随机顺序，
    # 使三组实验中每个batch的顺序也相同。
    dataloader_generator = torch.Generator().manual_seed(42)

    # 训练集使用shuffle，这叫小批量训练（mini-batch training）
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        generator=dataloader_generator
    )

    # 验证集只负责检查模型，不需要打乱
    validation_dataloader = DataLoader(
        validation_dataset,
        batch_size=64,
        shuffle=False
    )

    # 模型初始化：创建一个新的CNN，此时weight和bias还是初始值
    model = CifarCNN().to(device)

    # 损失函数：衡量10个类别分数和正确类别之间的差距
    loss_fn = nn.CrossEntropyLoss()

    # 优化器：根据梯度更新模型参数
    # lr是learning rate（学习率），决定参数每次更新的步幅
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    # 训练历史：保存每个epoch的指标，后面用来绘制训练曲线
    history = {
        "train_loss": [],
        "validation_accuracy": []
    }

    print(f"\n开始实验：learning_rate={learning_rate}")

    # 一个epoch表示完整遍历一次训练集
    for epoch in range(1, epochs + 1):
        # 训练模式：接下来要使用训练集更新参数
        model.train()

        train_loss_sum = 0.0
        train_total = 0

        # 每次取一个batch；一个batch完成一次参数更新
        for images, labels in train_dataloader:
            images = images.to(device)
            labels = labels.to(device)

            # 1. 前向传播：使用当前参数算出预测结果
            logits = model(images)

            # 2. 损失计算：比较预测结果和正确答案
            loss = loss_fn(logits, labels)

            # 3. 梯度清零：清除上一个batch留下的梯度
            optimizer.zero_grad()

            # 4. 反向传播：计算loss对每个参数的梯度
            loss.backward()

            # 5. 参数更新：Adam根据梯度和学习率修改参数
            optimizer.step()

            # 累计各个batch的损失，以便计算整个epoch的平均损失
            batch_size = len(labels)
            train_loss_sum += loss.item() * batch_size
            train_total += batch_size

        # 训练损失：这一整个epoch中，每张训练图片的平均损失
        train_loss = train_loss_sum / train_total

        # 验证准确率：衡量模型在未参与参数更新的数据上的表现
        validation_accuracy = evaluate_accuracy(
            model,
            validation_dataloader,
            device
        )

        # 将本轮指标追加到训练历史中
        history["train_loss"].append(train_loss)
        history["validation_accuracy"].append(
            validation_accuracy
        )

        print(
            f"第{epoch}轮："
            f"训练损失={train_loss:.4f}，"
            f"验证准确率={validation_accuracy * 100:.2f}%"
        )

    # 返回全部epoch的指标，而不只是最后一轮
    return history


# 本文件在仓库中的位置是code/2026-09-20/，向上三级到仓库根目录
project_folder = Path(__file__).resolve().parents[2]
data_folder = project_folder / "data"
artifact_folder = project_folder / "artifacts"
artifact_folder.mkdir(exist_ok=True)

# 数据预处理（preprocessing）：
# ToTensor把图片转成Tensor，Normalize调整像素数值范围
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5)
    )
])

source_dataset = datasets.CIFAR10(
    root=data_folder,
    train=True,
    download=False,
    transform=transform
)

# 数据划分（data split）：
# 固定随机种子后，三组实验使用完全相同的
# 10000张训练图片和2000张验证图片
split_generator = torch.Generator().manual_seed(42)
random_indices = torch.randperm(
    len(source_dataset),
    generator=split_generator
)

train_indices = random_indices[:10000]
validation_indices = random_indices[10000:12000]

train_dataset = Subset(source_dataset, train_indices)
validation_dataset = Subset(
    source_dataset,
    validation_indices
)

# 计算设备：优先使用GPU，没有可用GPU时使用CPU
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# 学习率是超参数；下面是本次要比较的三个候选取值
learning_rates = [0.0001, 0.001, 0.01]

# 实验结果：键是学习率，值是该学习率对应的训练历史
experiment_results = {}

print("使用设备：", device)
print("训练集大小：", len(train_dataset))
print("验证集大小：", len(validation_dataset))

# 依次完成三组实验
for learning_rate in learning_rates:
    history = run_experiment(
        learning_rate,
        train_dataset,
        validation_dataset,
        device
    )
    experiment_results[learning_rate] = history


# 可视化：创建一行两列的训练曲线图
figure, axes = plt.subplots(1, 2, figsize=(12, 4))

for learning_rate, history in experiment_results.items():
    epoch_numbers = range(
        1,
        len(history["train_loss"]) + 1
    )

    # 左图：训练损失随epoch的变化
    axes[0].plot(
        epoch_numbers,
        history["train_loss"],
        marker="o",
        label=f"lr={learning_rate}"
    )

    # 右图：验证准确率随epoch的变化
    axes[1].plot(
        epoch_numbers,
        history["validation_accuracy"],
        marker="o",
        label=f"lr={learning_rate}"
    )

axes[0].set_title("Training Loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend()
axes[0].grid(True)

axes[1].set_title("Validation Accuracy")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].legend()#每条线加上标签
axes[1].grid(True)#背景加网格

figure.tight_layout()

# 保存实验产物（artifact），方便以后对比实验
figure_path = artifact_folder / "learning_rate_curves.png"
figure.savefig(figure_path, dpi=150)#dpi是图片清晰度数据

print("\n训练曲线已保存到：", figure_path)
plt.show()
