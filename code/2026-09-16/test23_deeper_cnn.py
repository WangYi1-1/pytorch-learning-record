import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class DeeperFashionCNN(nn.Module):

    def __init__(self):
        super().__init__()

        # 第一层：1张灰度图 → 8张基础特征图
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=8,
            kernel_size=3,
            padding=1
        )
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)

        # 第二层：8张基础特征图 → 16张更复杂的特征图
        self.conv2 = nn.Conv2d(
            in_channels=8,
            out_channels=16,
            kernel_size=3,
            padding=1
        )
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)

        self.flatten = nn.Flatten()
        self.output_layer = nn.Linear(
            in_features=16 * 7 * 7,
            out_features=10
        )

    def forward(self, x):
        print("输入模型：", x.shape)

        x = self.conv1(x)
        x = self.relu1(x)
        print("第一层卷积后：", x.shape)

        x = self.pool1(x)
        print("第一次池化后：", x.shape)

        x = self.conv2(x)
        x = self.relu2(x)
        print("第二层卷积后：", x.shape)

        x = self.pool2(x)
        print("第二次池化后：", x.shape)

        x = self.flatten(x)
        print("展开后：", x.shape)

        logits = self.output_layer(x)
        print("输出类别分数：", logits.shape)

        return logits


dataset = datasets.FashionMNIST(
    root=r"D:\codexwork\pytorch学习\data",
    train=True,
    download=True,
    transform=transforms.ToTensor()
)

dataloader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True
)

images, labels = next(iter(dataloader))

model = DeeperFashionCNN()
logits = model(images)

print("第一层卷积权重：", model.conv1.weight.shape)
print("第二层卷积权重：", model.conv2.weight.shape)


