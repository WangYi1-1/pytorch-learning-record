"""指令、候选图片与历史融合：训练模型选择两步任务中的当前目标。

每个场景生成两条配对样本：指令和图片相同，历史与正确位置不同。
历史只记录已完成的第一步；本课先学习单步决策，不模拟移动或STOP。
"""

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

model_name = "google/bert_uncased_L-2_H-128_A-2"
color_names = ["red", "green", "blue", "yellow"]
color_values = torch.tensor([
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [1.0, 1.0, 0.0],
])

# 有序颜色组合：red→blue与blue→red是两条不同的指令。
color_pairs = [
    (first, second)
    for first in range(len(color_names))
    for second in range(len(color_names))
    if first != second
]
instruction_templates = [
    "first visit {first}, then visit {second}",
    "go to {first} first, then go to {second}",
    "first reach the {first} area, then reach the {second} area",
]


def make_noisy_color_image(color_value, generator, size=16):
    """与test45相同：生成带随机亮度和噪声的RGB图片。"""
    brightness = 0.7 + 0.3 * torch.rand(
        1, generator=generator
    ).item()
    image = color_value.view(3, 1, 1).expand(3, size, size).clone()
    noise = torch.randn((3, size, size), generator=generator) * 0.05
    return (image * brightness + noise).clamp(0.0, 1.0)


class HistoryNavigationDataset(Dataset):
    """每个场景保存历史为空、第一步已完成这两种状态。"""

    def __init__(self, scenario_count, seed):
        super().__init__()
        self.samples = []
        generator = torch.Generator().manual_seed(seed)

        for scenario_index in range(scenario_count):
            first_color_id, second_color_id = color_pairs[
                scenario_index % len(color_pairs)
            ]
            # 每个模板都覆盖全部颜色组合，避免句式与某个颜色绑定。
            template_index = (
                scenario_index // len(color_pairs)
            ) % len(instruction_templates)
            instruction = instruction_templates[template_index].format(
                first=color_names[first_color_id],
                second=color_names[second_color_id],
            )

            candidate_order = torch.randperm(
                len(color_names), generator=generator
            )
            candidate_images = torch.stack([
                make_noisy_color_image(color_values[color_id], generator)
                for color_id in candidate_order.tolist()
            ])
            candidate_names = [
                color_names[color_id]
                for color_id in candidate_order.tolist()
            ]

            # 历史编号：0=没有完成的任务，1～4=已完成的颜色编号+1。
            # 它只描述过去完成的任务，不直接提供当前正确候选位置。
            for history_id, target_color_id in [
                (0, first_color_id),
                (first_color_id + 1, second_color_id),
            ]:
                label = (
                    candidate_order == target_color_id
                ).nonzero(as_tuple=False).item()
                history_names = (
                    [] if history_id == 0
                    else [color_names[history_id - 1]]
                )
                self.samples.append({
                    "instruction": instruction,
                    "candidate_images": candidate_images,
                    "history_id": history_id,
                    "label": label,
                    "history_names": history_names,
                    "candidate_names": candidate_names,
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        return (
            sample["instruction"],
            sample["candidate_images"],
            sample["history_id"],
            sample["label"],
        )


class HistoryCandidateScorer(nn.Module):
    """文本编码器 + 图像编码器 + 历史Embedding + 候选打分器。"""

    def __init__(self, model_name, fusion_dim=16):
        super().__init__()
        self.text_encoder = AutoModel.from_pretrained(
            model_name, local_files_only=True
        )
        for parameter in self.text_encoder.parameters():
            parameter.requires_grad = False

        self.text_projection = nn.Linear(
            self.text_encoder.config.hidden_size, fusion_dim
        )
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),  # 卷积层
            nn.ReLU(),                              # 激活函数
            nn.MaxPool2d(2, 2),                      # 最大池化层
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),            # 全局平均池化
            nn.Flatten(),                            # 展平层
        )
        self.image_projection = nn.Linear(16, fusion_dim)

        # 历史编号[B] → 历史特征[B, 16]。
        # 0是一种有效历史状态，也有可训练向量；这里不设置padding_idx。
        self.history_embedding = nn.Embedding(
            num_embeddings=len(color_names) + 1,
            embedding_dim=fusion_dim,
        )
        # 上下文融合：完整指令特征 + 历史特征 → 当前决策特征。
        self.context_fusion = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim * 2),
            nn.ReLU(),
            nn.Linear(fusion_dim * 2, fusion_dim),
        )
        self.candidate_scorer = nn.Sequential(
            nn.Linear(fusion_dim * 3, fusion_dim * 2),
            nn.ReLU(),
            nn.Linear(fusion_dim * 2, 1),
        )

    def forward(
        self, input_ids, attention_mask, candidate_images,
        history_ids, token_type_ids=None,
    ):
        batch_size, candidate_count = candidate_images.shape[:2]
        text_outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        cls_features = text_outputs.last_hidden_state[:, 0, :]
        text_features = self.text_projection(cls_features)  # [B, 16]
        history_features = self.history_embedding(history_ids)  # [B, 16]
        context_features = self.context_fusion(torch.cat(
            [text_features, history_features], dim=-1
        ))  # [B, 32] → [B, 16]

        flat_images = candidate_images.reshape(
            batch_size * candidate_count, *candidate_images.shape[2:]
        )
        image_features = self.image_projection(
            self.image_encoder(flat_images)
        ).reshape(batch_size, candidate_count, -1)  # [B, 4, 16]

        expanded_context = context_features.unsqueeze(1).expand(
            batch_size, candidate_count, -1
        )


        interaction_features = expanded_context * image_features

        fused_features = torch.cat(
            [expanded_context, image_features, interaction_features], dim=-1
        )  # [B, 4, 48]
        return self.candidate_scorer(fused_features).squeeze(-1)  # [B, 4]


def evaluate(model, dataloader, loss_fn, device, use_history=True):
    """评估；use_history=False表示把全部历史替换为空历史。"""
    model.eval()
    loss_sum, correct_count, total_count = 0.0, 0, 0
    with torch.no_grad():
        for encoded, images, history_ids, labels in dataloader:
            encoded = encoded.to(device)
            images = images.to(device)
            history_ids = history_ids.to(device)
            labels = labels.to(device)
            if not use_history:
                history_ids = torch.zeros_like(history_ids)
            scores = model(
                candidate_images=images, history_ids=history_ids, **encoded
            )
            loss = loss_fn(scores, labels)
            batch_size = len(labels)
            loss_sum += loss.item() * batch_size
            correct_count += (scores.argmax(1) == labels).sum().item()
            total_count += batch_size
    return loss_sum / total_count, correct_count / total_count


def main():
    torch.manual_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, local_files_only=True
    )

    def collate_batch(batch):
        instructions, images, history_ids, labels = zip(*batch)
        encoded = tokenizer(
            list(instructions), padding=True, truncation=True,
            max_length=32, return_tensors="pt",
        )
        return (
            encoded,
            torch.stack(images),                       # [B, 4, 3, 16, 16]
            torch.tensor(history_ids, dtype=torch.long),  # [B]
            torch.tensor(labels, dtype=torch.long),       # [B]
        )

    # 相同句式与颜色组合，独立生成图片噪声和候选顺序。
    # 每个场景在本数据集内部成对保存，不跨训练/验证/测试拆分。
    train_dataset = HistoryNavigationDataset(scenario_count=180, seed=11)
    validation_dataset = HistoryNavigationDataset(scenario_count=36, seed=12)
    test_dataset = HistoryNavigationDataset(scenario_count=36, seed=13)
    train_dataloader = DataLoader(
        train_dataset, batch_size=32, shuffle=True, collate_fn=collate_batch
    )
    validation_dataloader = DataLoader(
        validation_dataset, batch_size=32, shuffle=False,
        collate_fn=collate_batch,
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=32, shuffle=False, collate_fn=collate_batch
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HistoryCandidateScorer(model_name).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=0.003
    )

    model_folder = Path(__file__).resolve().parent.parent / "models"
    model_folder.mkdir(exist_ok=True)
    model_path = model_folder / "best_history_candidate_scorer.pth"
    best_accuracy, best_loss = -1.0, float("inf")
    epochs = 40
    print("设备：", device)
    print(
        "训练/验证/测试样本数：",
        len(train_dataset), len(validation_dataset), len(test_dataset),
    )

    for epoch in range(1, epochs + 1):
        model.train()
        model.text_encoder.eval()  # 冻结的BERT保持评估模式，关闭Dropout。
        loss_sum, correct_count, total_count = 0.0, 0, 0
        for encoded, images, history_ids, labels in train_dataloader:
            encoded = encoded.to(device)
            images = images.to(device)
            history_ids = history_ids.to(device)
            labels = labels.to(device)
            scores = model(
                candidate_images=images, history_ids=history_ids, **encoded
            )
            loss = loss_fn(scores, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_size = len(labels)
            loss_sum += loss.item() * batch_size
            correct_count += (scores.argmax(1) == labels).sum().item()
            total_count += batch_size

        validation_loss, validation_accuracy = evaluate(
            model, validation_dataloader, loss_fn, device
        )
        if (
            validation_accuracy > best_accuracy
            or (validation_accuracy == best_accuracy and validation_loss < best_loss)
        ):
            best_accuracy, best_loss = validation_accuracy, validation_loss
            torch.save(model.state_dict(), model_path)
        if epoch == 1 or epoch % 5 == 0:
            print(
                f"第{epoch:2d}轮 | 训练损失={loss_sum / total_count:.4f} | "
                f"训练准确率={correct_count / total_count:.2%} | "
                f"验证准确率={validation_accuracy:.2%}"
            )

    model.load_state_dict(torch.load(
        model_path, map_location=device, weights_only=True
    ))
    test_loss, test_accuracy = evaluate(model, test_dataloader, loss_fn, device)
    _, no_history_accuracy = evaluate(
        model, test_dataloader, loss_fn, device, use_history=False
    )
    print(f"\n正常历史：测试损失={test_loss:.4f}，准确率={test_accuracy:.2%}")
    print(f"历史全部清空：准确率={no_history_accuracy:.2%}")
    print("配对样本在清空历史后输入完全相同、标签不同，最高只能答对一半。")

    # 配对预测展示：取同一场景中两条现成样本，指令和图片相同、历史不同。
    # 这里不修改历史；观察模型是否按历史状态切换目标（正确标签也不同）。
    pair = test_dataset.samples[:2]
    batch = collate_batch([test_dataset[0], test_dataset[1]])
    encoded, images, history_ids, _ = batch
    model.eval()
    with torch.no_grad():
        predictions = model(
            candidate_images=images.to(device),
            history_ids=history_ids.to(device),
            **encoded.to(device),
        ).argmax(1).cpu().tolist()
    print("\n同一场景的历史对照：")
    print("指令：", pair[0]["instruction"])
    print("候选：", pair[0]["candidate_names"])
    for sample, prediction in zip(pair, predictions):
        print(
            f"历史={sample['history_names']} | "
            f"正确位置={sample['label']} | 预测位置={prediction}"
        )
    print("\n最佳模型：", model_path)
    print("本实验仅验证受控两步任务中的历史利用，未测试陌生句式和真实环境。")


if __name__ == "__main__":
    main()
