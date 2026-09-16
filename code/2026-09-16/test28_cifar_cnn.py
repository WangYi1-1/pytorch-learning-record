import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class CifarCNN(nn.Module):

    def __init__(self):
        super().__init__()

        # 连续执行的卷积层打包成特征提取器
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        # 连续执行的全连接层打包成分类器
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


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5)
    )
])

dataset = datasets.CIFAR10(
    root=r"D:\codexwork\pytorch学习\data",
    train=True,
    download=False,
    transform=transform
)

dataloader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True
)

images, labels = next(iter(dataloader))

model = CifarCNN()

# 分开调用两部分，方便观察中间形状
feature_maps = model.features(images)
logits = model.classifier(feature_maps)

print(model)
print("输入图片形状：", images.shape)
print("特征提取后的形状：", feature_maps.shape)
print("最终类别分数形状：", logits.shape)


