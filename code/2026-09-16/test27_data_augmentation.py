import matplotlib.pyplot as plt
from torchvision import datasets, transforms


train_transform = transforms.Compose([
    # 图片外围补4格，再随机裁剪回32×32，模拟轻微位置变化
    transforms.RandomCrop(32, padding=4),

    # 每次有50%的概率进行左右翻转
    transforms.RandomHorizontalFlip(p=0.5),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5)
    )
])

train_dataset = datasets.CIFAR10(
    root=r"D:\codexwork\pytorch学习\data",
    train=True,
    download=False,
    transform=train_transform
)


figure, axes = plt.subplots(1, 5, figsize=(15, 3))

for index in range(5):
    # 每次读取同一个编号，都会重新执行一次随机数据增强
    image, label = train_dataset[0]

    # 将[-1,1]还原到[0,1]，只用于正确显示图片
    display_image = image * 0.5 + 0.5

    # PyTorch的[C,H,W]改成Matplotlib需要的[H,W,C]
    display_image = display_image.permute(1, 2, 0)

    axes[index].imshow(display_image)
    axes[index].set_title(train_dataset.classes[label])
    axes[index].axis("off")

plt.tight_layout()
plt.show()


