import os
import sys

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer
from transformers.utils import logging


sys.stdout.reconfigure(encoding="utf-8")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
logging.set_verbosity_error()
torch.manual_seed(42)


model_name = "google/bert_uncased_L-2_H-128_A-2"
candidate_count = 4

# 两条导航指令，每条指令同时配有4张候选方向图片。
instructions = [
    "move toward the red area",
    "go toward the blue area"
]


def make_color_image(red, green, blue, size=16):
    """创建一张简单的RGB纯色图片，形状为[3, height, width]。"""
    color = torch.tensor(
        [red, green, blue],
        dtype=torch.float32
    )

    return color.view(3, 1, 1).expand(
        3,
        size,
        size
    ).clone()


red_image = make_color_image(1.0, 0.0, 0.0)
green_image = make_color_image(0.0, 1.0, 0.0)
blue_image = make_color_image(0.0, 0.0, 1.0)
yellow_image = make_color_image(1.0, 1.0, 0.0)

# 第1条指令的红色图在第0个候选位置。
# 第2条指令的蓝色图在第3个候选位置。
candidate_images = torch.stack([
    torch.stack([
        red_image,
        green_image,
        blue_image,
        yellow_image
    ]),
    torch.stack([
        yellow_image,
        red_image,
        green_image,
        blue_image
    ])
])

correct_candidate_indices = torch.tensor([
    0,
    3
])


class MultimodalCandidateScorer(nn.Module):
    """用文本指令和候选图片共同为每个候选方向打分。"""

    def __init__(self, model_name, fusion_dim):
        super().__init__()

        # 文本骨干：预训练BERT。
        self.text_encoder = AutoModel.from_pretrained(
            model_name,
            local_files_only=True
        )

        for parameter in self.text_encoder.parameters():
            parameter.requires_grad = False

        text_feature_count = (
            self.text_encoder.config.hidden_size
        )#读维度

        # 图像骨干：小型CNN，每张候选图片输出16维特征。
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

        # 把BERT的128维文本特征投影到16维融合空间。
        self.text_projection = nn.Linear(
            text_feature_count,
            fusion_dim
        )

        # CNN已经输出16维，这一层让它也进入可学习的融合空间。
        self.image_projection = nn.Linear(
            16,
            fusion_dim
        )

        # 拼接后特征是16+16=32维，最后输出该候选方向的一个分数。
        self.candidate_scorer = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
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

        # -------- 文本分支 --------
        text_outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        # [batch, sequence, 128] → [batch, 128]
        cls_features = (
            text_outputs.last_hidden_state[:, 0, :]
        )

        # [batch, 128] → [batch, 16]
        text_features = self.text_projection(
            cls_features
        )

        # -------- 图像分支 --------
        # [batch, candidates, channels, height, width]
        # → [batch*candidates, channels, height, width]
        flat_candidate_images = candidate_images.reshape(
            batch_size * candidate_count,
            *candidate_images.shape[2:]
        )

        # [batch*candidates, 3, 16, 16]
        # → [batch*candidates, 16]
        flat_image_features = self.image_encoder(
            flat_candidate_images
        )#CNN层

        flat_image_features = self.image_projection(
            flat_image_features
        )#投影层（Linear）

        # [batch*candidates, 16]
        # → [batch, candidates, 16]
        image_features = flat_image_features.reshape(
            batch_size,
            candidate_count,
            -1
        )

        # -------- 图文融合 --------
        # 每条指令要与它的4张候选图分别比较，
        # 因此将[batch,16]扩展成[batch,candidates,16]。
        expanded_text_features = text_features.unsqueeze(1).expand(
            batch_size,
            candidate_count,
            -1
        )

        # 拼接文本和图像特征：
        # [batch, candidates, 16+16]
        fused_features = torch.cat(
            [expanded_text_features, image_features],
            dim=-1
        )

        # 每个候选方向得到1个logit：
        # [batch, candidates, 32]
        # → [batch, candidates, 1]
        # → [batch, candidates]
        candidate_scores = self.candidate_scorer(
            fused_features
        ).squeeze(-1)

        return (
            candidate_scores,
            text_features,
            image_features,
            fused_features
        )


tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    local_files_only=True
)

encoded_inputs = tokenizer(
    instructions,
    padding=True,
    truncation=True,
    return_tensors="pt"
)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = MultimodalCandidateScorer(
    model_name=model_name,
    fusion_dim=16
).to(device)

encoded_inputs = encoded_inputs.to(device)
candidate_images = candidate_images.to(device)

model.eval()

with torch.no_grad():
    (
        candidate_scores,
        text_features,
        image_features,
        fused_features
    ) = model(
        candidate_images=candidate_images,
        **encoded_inputs
    )

predicted_candidate_indices = candidate_scores.argmax(
    dim=1
).cpu()

print("指令数量：", len(instructions))
print("每条指令的候选图片数：", candidate_count)
print("candidate_images形状：", candidate_images.shape)
print("text_features形状：", text_features.shape)
print("image_features形状：", image_features.shape)
print("fused_features形状：", fused_features.shape)
print("candidate_scores形状：", candidate_scores.shape)

print("\n每个候选方向的分数：")
print(candidate_scores)

print("\n正确候选下标：", correct_candidate_indices)
print("当前预测下标：", predicted_candidate_indices)
print(
    "\n注意：CNN、投影层和打分器还没有训练，"
    "因此当前分数和预测没有任务意义。"
)
