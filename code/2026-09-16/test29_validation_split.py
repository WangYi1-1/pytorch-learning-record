import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


# 训练集可以使用随机数据增强
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5)
    )
])

# 验证集和测试集不使用随机数据增强
evaluation_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5)
    )
])

# 同一份50000张训练图片，创建两个transform不同的dataset对象
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

# 固定随机种子，让每次运行得到相同的划分
generator = torch.Generator().manual_seed(42)
random_indices = torch.randperm(
    len(train_source),
    generator=generator
)

# 前45000个随机编号用于训练，剩余5000个用于验证
train_indices = random_indices[:45000]
validation_indices = random_indices[45000:]

train_dataset = Subset(train_source, train_indices)
validation_dataset = Subset(
    validation_source,
    validation_indices
)

# 官方10000张测试图片只留到最后测试
test_dataset = datasets.CIFAR10(
    root=r"D:\codexwork\pytorch学习\data",
    train=False,
    download=False,
    transform=evaluation_transform
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

train_images, train_labels = next(iter(train_dataloader))
validation_images, validation_labels = next(
    iter(validation_dataloader)
)
test_images, test_labels = next(iter(test_dataloader))

print("训练集图片数量：", len(train_dataset))
print("验证集图片数量：", len(validation_dataset))
print("测试集图片数量：", len(test_dataset))

print("训练批次形状：", train_images.shape, train_labels.shape)
print(
    "验证批次形状：",
    validation_images.shape,
    validation_labels.shape
)
print("测试批次形状：", test_images.shape, test_labels.shape)


