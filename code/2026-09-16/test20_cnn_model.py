import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class FashionCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=4,
            kernel_size=3,
            padding=1
        )

        self.relu = nn.ReLU()

        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        self.flatten = nn.Flatten()

        self.output_layer = nn.Linear(
            in_features=4 * 14 * 14,
            out_features=10
        )

    def forward(self, x):
        x = self.conv(x)          # [N, 1, 28, 28] → [N, 4, 28, 28]
        x = self.relu(x)          # 形状不变
        x = self.pool(x)          # [N, 4, 28, 28] → [N, 4, 14, 14]
        x = self.flatten(x)       # [N, 4, 14, 14] → [N, 784]
        logits = self.output_layer(x)  # [N, 784] → [N, 10]
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

model = FashionCNN()
logits = model(images)

print(model)
print("输入图片形状：", images.shape)
print("类别分数形状：", logits.shape)
print("卷积层权重形状：", model.conv.weight.shape)
print("输出层权重形状：", model.output_layer.weight.shape)


