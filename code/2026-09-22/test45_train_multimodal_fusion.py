import os
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer
from transformers.utils import logging


sys.stdout.reconfigure(encoding="utf-8")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
logging.set_verbosity_error()
torch.manual_seed(42)


model_name = "google/bert_uncased_L-2_H-128_A-2"

color_names = [
    "red",
    "green",
    "blue",
    "yellow"
]

color_values = torch.tensor([
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [1.0, 1.0, 0.0]
])


def make_noisy_color_image(
    color_value,
    generator,
    size=16
):
    """生成带随机亮度和噪声的颜色图片。"""
    brightness = 0.7 + 0.3 * torch.rand(
        1,
        generator=generator
    ).item()

    image = color_value.view(3, 1, 1).expand(
        3,
        size,
        size
    ).clone()

    noise = torch.randn(
        (3, size, size),
        generator=generator
    ) * 0.05

    return (image * brightness + noise).clamp(0.0, 1.0)


class ColorNavigationDataset(Dataset):
    """自动生成“文字颜色指令 + 4张随机顺序候选图”。"""

    def __init__(self, sample_count, templates, seed):
        super().__init__()
        self.samples = []
        generator = torch.Generator().manual_seed(seed)

        for sample_index in range(sample_count):
            target_color_index = sample_index % len(color_names)
            target_color_name = color_names[target_color_index]

            template = templates[
                sample_index % len(templates)
            ]
            instruction = template.format(
                color=target_color_name
            )

            # 将四种颜色的候选顺序随机打乱。
            candidate_order = torch.randperm(
                len(color_names),
                generator=generator
            )

            candidate_images = torch.stack([
                make_noisy_color_image(
                    color_values[color_index],
                    generator
                )
                for color_index in candidate_order.tolist()
            ])

            # 找到目标颜色在当前乱序候选图中的位置。
            correct_candidate_index = (
                candidate_order == target_color_index
            ).nonzero(as_tuple=False).item()

            candidate_color_names = [
                color_names[color_index]
                for color_index in candidate_order.tolist()
            ]

            self.samples.append({
                "instruction": instruction,
                "candidate_images": candidate_images,
                "correct_candidate_index": (
                    correct_candidate_index
                ),
                "candidate_color_names": (
                    candidate_color_names
                )
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        return (
            sample["instruction"],
            sample["candidate_images"],
            sample["correct_candidate_index"]
        )


tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    local_files_only=True
)


def collate_batch(batch):
    instructions, candidate_images, labels = zip(*batch)

    encoded_inputs = tokenizer(
        list(instructions),
        padding=True,
        truncation=True,
        max_length=32,
        return_tensors="pt"
    )

    candidate_images = torch.stack(candidate_images)
    labels = torch.tensor(labels, dtype=torch.long)

    return encoded_inputs, candidate_images, labels


class MultimodalCandidateScorer(nn.Module):

    def __init__(self, model_name, fusion_dim):
        super().__init__()

        self.text_encoder = AutoModel.from_pretrained(
            model_name,
            local_files_only=True
        )

        for parameter in self.text_encoder.parameters():
            parameter.requires_grad = False

        text_feature_count = (
            self.text_encoder.config.hidden_size
        )

        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

        self.text_projection = nn.Linear(
            text_feature_count,
            fusion_dim
        )
        self.image_projection = nn.Linear(
            16,
            fusion_dim
        )

        self.candidate_scorer = nn.Sequential(
            nn.Linear(fusion_dim * 3, fusion_dim),
            nn.ReLU(),
            nn.Linear(fusion_dim, 1)
        )

    def forward(
        self,
        input_ids,
        attention_mask,
        candidate_images,
        token_type_ids=None
    ):
        batch_size = candidate_images.shape[0]
        candidate_count = candidate_images.shape[1]

        text_outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        cls_features = (
            text_outputs.last_hidden_state[:, 0, :]
        )
        text_features = self.text_projection(cls_features)

        flat_candidate_images = candidate_images.reshape(
            batch_size * candidate_count,
            *candidate_images.shape[2:]
        )

        flat_image_features = self.image_encoder(
            flat_candidate_images
        )
        flat_image_features = self.image_projection(
            flat_image_features
        )

        image_features = flat_image_features.reshape(
            batch_size,
            candidate_count,
            -1
        )

        expanded_text_features = text_features.unsqueeze(1).expand(
            batch_size,
            candidate_count,
            -1
        )

        # 逐元素乘法显式表示“文字和图像特征在哪些维度同时激活”。
        interaction_features = (
            expanded_text_features * image_features
        )

        # 16维文字 + 16维图像 + 16维交互 = 48维融合特征。
        fused_features = torch.cat(
            [
                expanded_text_features,
                image_features,
                interaction_features
            ],
            dim=-1
        )

        return self.candidate_scorer(
            fused_features
        ).squeeze(-1)


def evaluate(model, dataloader, loss_fn, device):
    model.eval()

    loss_sum = 0.0
    correct_count = 0
    total_count = 0

    with torch.no_grad():
        for encoded_inputs, candidate_images, labels in dataloader:
            encoded_inputs = encoded_inputs.to(device)
            candidate_images = candidate_images.to(device)
            labels = labels.to(device)

            candidate_scores = model(
                candidate_images=candidate_images,
                **encoded_inputs
            )
            loss = loss_fn(candidate_scores, labels)

            batch_size = len(labels)
            loss_sum += loss.item() * batch_size
            correct_count += (
                candidate_scores.argmax(dim=1) == labels
            ).sum().item()
            total_count += batch_size

    return (
        loss_sum / total_count,
        correct_count / total_count
    )


def print_predictions(model, dataset, device, count=8):
    model.eval()

    print("\n部分测试样本：")

    with torch.no_grad():
        for sample in dataset.samples[:count]:
            encoded_inputs = tokenizer(
                [sample["instruction"]],
                return_tensors="pt"
            ).to(device)

            candidate_images = sample[
                "candidate_images"
            ].unsqueeze(0).to(device)

            candidate_scores = model(
                candidate_images=candidate_images,
                **encoded_inputs
            )

            predicted_index = candidate_scores.argmax(
                dim=1
            ).item()

            correct_index = sample[
                "correct_candidate_index"
            ]
            candidate_names = sample[
                "candidate_color_names"
            ]

            print(
                f"指令={sample['instruction']:<35} | "
                f"候选={candidate_names} | "
                f"正确={correct_index} | "
                f"预测={predicted_index}"
            )


instruction_templates = [
    "move toward the {color} area",
    "go to the {color} region",
    "head toward {color}",
    "choose the {color} direction"
]

train_dataset = ColorNavigationDataset(
    sample_count=120,
    templates=instruction_templates,
    seed=1
)

validation_dataset = ColorNavigationDataset(
    sample_count=40,
    templates=instruction_templates,
    seed=2
)

test_dataset = ColorNavigationDataset(
    sample_count=40,
    templates=instruction_templates,
    seed=3
)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    collate_fn=collate_batch
)
validation_dataloader = DataLoader(
    validation_dataset,
    batch_size=16,
    shuffle=False,
    collate_fn=collate_batch
)
test_dataloader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False,
    collate_fn=collate_batch
)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = MultimodalCandidateScorer(
    model_name=model_name,
    fusion_dim=16
).to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ],
    lr=0.003
)

project_folder = Path(__file__).resolve().parent.parent
model_folder = project_folder / "models"
model_folder.mkdir(exist_ok=True)
model_path = model_folder / "best_multimodal_candidate_scorer.pth"

epochs = 30
best_validation_accuracy = 0.0
best_validation_loss = float("inf")

print("使用设备：", device)
print("训练样本数：", len(train_dataset))
print("验证样本数：", len(validation_dataset))
print("测试样本数：", len(test_dataset))

for epoch in range(1, epochs + 1):
    model.train()
    model.text_encoder.eval()

    train_loss_sum = 0.0
    train_correct = 0
    train_total = 0

    for encoded_inputs, candidate_images, labels in train_dataloader:
        encoded_inputs = encoded_inputs.to(device)
        candidate_images = candidate_images.to(device)
        labels = labels.to(device)

        candidate_scores = model(
            candidate_images=candidate_images,
            **encoded_inputs
        )
        loss = loss_fn(candidate_scores, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = len(labels)
        train_loss_sum += loss.item() * batch_size
        train_correct += (
            candidate_scores.argmax(dim=1) == labels
        ).sum().item()
        train_total += batch_size

    train_loss = train_loss_sum / train_total
    train_accuracy = train_correct / train_total

    validation_loss, validation_accuracy = evaluate(
        model,
        validation_dataloader,
        loss_fn,
        device
    )

    is_better_model = (
        validation_accuracy > best_validation_accuracy
        or (
            validation_accuracy == best_validation_accuracy
            and validation_loss < best_validation_loss
        )
    )

    if is_better_model:
        best_validation_accuracy = validation_accuracy
        best_validation_loss = validation_loss

        torch.save(
            model.state_dict(),
            model_path
        )

    if epoch == 1 or epoch % 5 == 0:
        print(
            f"第{epoch:2d}轮 | "
            f"训练损失={train_loss:.4f} | "
            f"训练准确率={train_accuracy * 100:6.2f}% | "
            f"验证准确率={validation_accuracy * 100:6.2f}%"
        )


best_model = MultimodalCandidateScorer(
    model_name=model_name,
    fusion_dim=16
).to(device)

best_model.load_state_dict(
    torch.load(
        model_path,
        map_location=device,
        weights_only=True
    )
)

test_loss, test_accuracy = evaluate(
    best_model,
    test_dataloader,
    loss_fn,
    device
)

print("\n图文融合模型的测试结果：")
print(f"测试损失：{test_loss:.4f}")
print(f"测试准确率：{test_accuracy * 100:.2f}%")

print_predictions(
    best_model,
    test_dataset,
    device
)

print("\n最佳模型已保存到：", model_path)
