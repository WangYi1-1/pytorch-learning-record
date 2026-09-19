from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class CifarCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

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


# 本文件在仓库中的位置是 code/2026-09-19/，因此向上三级到仓库根目录
project_folder = Path(__file__).resolve().parents[2]
data_folder = project_folder / "data"
model_path = project_folder / "models" / "best_cifar_cnn.pth"

evaluation_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5)
    )
])

test_dataset = datasets.CIFAR10(
    root=data_folder,
    train=False,
    download=False,
    transform=evaluation_transform
)

test_dataloader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = CifarCNN().to(device)
model.load_state_dict(
    torch.load(
        model_path,
        map_location=device,
        weights_only=True
    )
)
model.eval()

class_count = len(test_dataset.classes)
class_correct = torch.zeros(class_count, dtype=torch.long)
class_total = torch.zeros(class_count, dtype=torch.long)

# 行表示正确类别，列表示模型预测类别
confusion_matrix = torch.zeros(
    class_count,
    class_count,
    dtype=torch.long
)

mistakes = []

with torch.no_grad():
    for images, labels in test_dataloader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        predictions = logits.argmax(dim=1)

        cpu_images = images.cpu()
        cpu_labels = labels.cpu()
        cpu_predictions = predictions.cpu()

        for index in range(len(cpu_labels)):
            correct_label = cpu_labels[index].item()
            predicted_label = cpu_predictions[index].item()

            class_total[correct_label] += 1
            confusion_matrix[correct_label, predicted_label] += 1

            if correct_label == predicted_label:
                class_correct[correct_label] += 1
            elif len(mistakes) < 8:
                mistakes.append((
                    cpu_images[index],
                    correct_label,
                    predicted_label
                ))

total_correct = class_correct.sum().item()
total_count = class_total.sum().item()
overall_accuracy = total_correct / total_count

print("使用设备：", device)
print("测试集整体准确率：", f"{overall_accuracy * 100:.2f}%")
print("\n每个类别的准确率：")

for class_index, class_name in enumerate(test_dataset.classes):
    accuracy = (
        class_correct[class_index].item()
        / class_total[class_index].item()
    )
    print(f"{class_name:>10s}：{accuracy * 100:6.2f}%")

print("\n混淆矩阵：")
print("行=正确类别，列=预测类别")
print(confusion_matrix)

figure, axes = plt.subplots(2, 4, figsize=(12, 6))

for axis, (image, correct_label, predicted_label) in zip(
    axes.flat,
    mistakes
):
    # 撤销 Normalize，将像素范围从 [-1, 1] 还原到 [0, 1]
    display_image = image * 0.5 + 0.5
    display_image = display_image.permute(1, 2, 0)

    axis.imshow(display_image)
    axis.set_title(
        f"True: {test_dataset.classes[correct_label]}\n"
        f"Predicted: {test_dataset.classes[predicted_label]}"
    )
    axis.axis("off")

figure.suptitle("Misclassified CIFAR-10 Images")
figure.tight_layout()
plt.show()
