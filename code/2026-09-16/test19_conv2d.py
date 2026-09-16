import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


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

# 输入1个通道，使用4个3×3的卷积核
conv = nn.Conv2d(
    in_channels=1,
    out_channels=4,
    kernel_size=3,
    padding=1
)

relu = nn.ReLU()

# 每张特征图的长和宽都缩小为原来的一半
pool = nn.MaxPool2d(
    kernel_size=2,
    stride=2
)

feature_maps = conv(images)
activated_maps = relu(feature_maps)
pooled_maps = pool(activated_maps)

print("原始图片形状：", images.shape)
print("卷积核权重形状：", conv.weight.shape)
print("卷积后形状：", feature_maps.shape)
print("ReLU后形状：", activated_maps.shape)
print("池化后形状：", pooled_maps.shape)


