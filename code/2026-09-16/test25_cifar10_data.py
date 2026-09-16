import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


dataset = datasets.CIFAR10(
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

print("CIFAR-10训练图片数量：", len(dataset))
print("一批图片形状：", images.shape)
print("一批标签形状：", labels.shape)
print("10个类别名称：", dataset.classes)


# 取这一批中的第一张彩色图片
image = images[0]
label = labels[0].item()

print("单张图片原始形状：", image.shape)

# PyTorch使用[C, H, W]，Matplotlib使用[H, W, C]
display_image = image.permute(1, 2, 0)

print("调整显示顺序后：", display_image.shape)
print("这张图片的类别：", dataset.classes[label])

plt.imshow(display_image)
plt.title(dataset.classes[label])
plt.axis("off")
plt.show()


