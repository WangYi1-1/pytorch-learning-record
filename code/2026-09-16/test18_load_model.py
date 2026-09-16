from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class ImageMLP(nn.Module):

    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layer1 = nn.Linear(784, 128)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.layer1(x)
        x = self.relu(x)
        logits = self.layer2(x)
        return logits


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model_path = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "fashion_mlp.pth"
)

if not model_path.exists():
    raise FileNotFoundError(
        "还没有模型参数文件，请先运行 test17_evaluate.py"
    )

# 先创建一个结构相同、参数随机的新模型
model = ImageMLP().to(device)

# 把保存的参数装入新模型
model.load_state_dict(torch.load(
    model_path,
    map_location=device,
    weights_only=True
))
model.eval()


test_dataset = datasets.FashionMNIST(
    root=r"D:\codexwork\pytorch学习\data",
    train=False,
    download=True,
    transform=transforms.ToTensor()
)

test_dataloader = DataLoader(
    test_dataset,
    batch_size=8,
    shuffle=False
)

images, labels = next(iter(test_dataloader))
images = images.to(device)

with torch.no_grad():
    logits = model(images)
    predicted_classes = logits.argmax(dim=1).cpu()

print("前8张测试图片的预测结果：")

for index in range(len(labels)):
    correct_name = test_dataset.classes[labels[index].item()]
    predicted_name = test_dataset.classes[
        predicted_classes[index].item()
    ]

    print(
        f"第{index + 1}张："
        f"正确类别={correct_name}，"
        f"预测类别={predicted_name}"
    )


