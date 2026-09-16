from torchvision import datasets, transforms


# 只转换成Tensor：像素范围从0～255变成0～1
original_transform = transforms.ToTensor()

# 按顺序执行ToTensor和Normalize
normalized_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5)
    )
])

original_dataset = datasets.CIFAR10(
    root=r"D:\codexwork\pytorch学习\data",
    train=True,
    download=False,
    transform=original_transform
)

normalized_dataset = datasets.CIFAR10(
    root=r"D:\codexwork\pytorch学习\data",
    train=True,
    download=False,
    transform=normalized_transform
)

# 两个dataset中的第0张是同一张图片，只是transform不同
original_image, original_label = original_dataset[0]
normalized_image, normalized_label = normalized_dataset[0]

print("两次读取的标签：", original_label, normalized_label)
print("图片形状：", original_image.shape)

print(
    "处理前的数值范围：",
    original_image.min().item(),
    "到",
    original_image.max().item()
)

print(
    "标准化后的数值范围：",
    normalized_image.min().item(),
    "到",
    normalized_image.max().item()
)

print("处理前第一个像素的RGB：", original_image[:, 0, 0])
print("标准化后同一像素的RGB：", normalized_image[:, 0, 0])


