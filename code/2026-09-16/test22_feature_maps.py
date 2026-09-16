from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn
from torchvision import datasets, transforms


class FashionCNN(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.output_layer = nn.Linear(4 * 14 * 14, 10)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.flatten(x)
        logits = self.output_layer(x)
        return logits


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model_path = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "fashion_cnn.pth"
)

model = FashionCNN().to(device)
model.load_state_dict(
    torch.load(
        model_path,
        map_location=device,
        weights_only=True
    )
)
model.eval()


test_dataset = datasets.FashionMNIST(
    root=r"D:\codexwork\pytorch学习\data",
    train=False,
    download=True,
    transform=transforms.ToTensor()
)

# dataset[0]返回一张图片和它的正确类别编号
image, label = test_dataset[0]

# 单张图片是[C, H, W]，增加批次维度后成为[1, C, H, W]
image_batch = image.unsqueeze(0).to(device)

with torch.no_grad():
    feature_maps = model.conv(image_batch)
    activated_maps = model.relu(feature_maps)
    logits = model(image_batch)
    predicted_class = logits.argmax(dim=1).item()

print("单张图片形状：", image.shape)
print("增加批次维度后：", image_batch.shape)
print("卷积产生的特征图：", feature_maps.shape)
print("正确类别：", test_dataset.classes[label])
print("预测类别：", test_dataset.classes[predicted_class])


# 第一幅显示原图，后四幅显示4个卷积核得到的特征图
figure, axes = plt.subplots(1, 5, figsize=(15, 3))

axes[0].imshow(image[0], cmap="gray")
axes[0].set_title("Original image")
axes[0].axis("off")

for index in range(4):
    feature_map = activated_maps[0, index].cpu()

    axes[index + 1].imshow(feature_map, cmap="gray")
    axes[index + 1].set_title(f"Feature map {index + 1}")
    axes[index + 1].axis("off")

plt.tight_layout()
plt.show()


