import torch
from torch import nn


class LinearModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.tensor(0.0))
        self.b = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        return self.w * x + self.b


inputs = torch.tensor([
    [-3.0],
    [-2.0],
    [-1.0],
    [0.0],
    [1.0],
    [2.0],
    [3.0]
])

targets = 2 * inputs + 1
loss_fn = nn.MSELoss()


def train_model(optimizer_name):
    # 每次调用都创建同样的模型，初始参数都是 w=0、b=0
    model = LinearModel()

    if optimizer_name == "SGD":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=0.05
        )
    elif optimizer_name == "Adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=0.05
        )
    else:
        raise ValueError("不支持的优化器")

    recorded_losses = {}

    for epoch in range(1, 61):
        predictions = model(inputs)
        loss = loss_fn(predictions, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch in [1, 5, 10, 20, 40, 60]:
            recorded_losses[epoch] = loss.item()

    return model, recorded_losses


sgd_model, sgd_losses = train_model("SGD")
adam_model, adam_losses = train_model("Adam")

print("轮数      SGD损失      Adam损失")

for epoch in sgd_losses:
    print(
        f"{epoch:>4d}    "
        f"{sgd_losses[epoch]:>10.6f}    "
        f"{adam_losses[epoch]:>10.6f}"
    )

print("\nSGD训练后的参数：")
print("w =", sgd_model.w.item())
print("b =", sgd_model.b.item())

print("\nAdam训练后的参数：")
print("w =", adam_model.w.item())
print("b =", adam_model.b.item())
