"""第48课：把单步预测接成连续执行的两步任务。

闭环执行（rollout）：读取当前状态 → 模型选候选 → 执行 → 验证 → 更新历史。
复用第47课的模型和参数，不重新训练，也不覆盖原来的模型文件。

这是静态候选图片上的玩具环境，不模拟三维移动。
环境用已知任务规则确认子任务是否完成；STOP也由规则决定，不由模型学习。
正确标签只供环境检查完成情况，绝不作为模型输入或替模型选择动作。
"""

from pathlib import Path

import torch
from transformers import AutoTokenizer

from test47_train_with_history import (
    HistoryCandidateScorer,
    HistoryNavigationDataset,
    color_names,
    model_name,
)


class ToyNavigationEnvironment:
    """玩具任务环境：保存场景、目标顺序与实际完成的历史。"""

    def __init__(self, first_sample, second_sample):
        # 两条样本来自同一场景，文字与候选图片相同。
        self.instruction = first_sample["instruction"]
        self.candidate_images = first_sample["candidate_images"]
        self.candidate_names = first_sample["candidate_names"]
        # 环境持有任务真值，仅用于检查进度；模型看不到subtasks。
        self.subtasks = [
            self.candidate_names[first_sample["label"]],
            self.candidate_names[second_sample["label"]],
        ]
        self.completed_subtasks = []

    def history_id(self):
        """历史编码：继续使用第47课的0=空历史、颜色编号+1约定。"""
        if not self.completed_subtasks:
            return 0
        return color_names.index(self.completed_subtasks[0]) + 1

    def is_done(self):
        """终止条件：两个子任务都已按顺序完成。"""
        return len(self.completed_subtasks) == len(self.subtasks)

    def step(self, candidate_index):
        """执行动作并检查结果；只有完成当前目标，才推进任务进度。"""
        if self.is_done():
            raise RuntimeError("任务已完成，不应再执行候选动作。")
        chosen_color = self.candidate_names[candidate_index]
        current_target = self.subtasks[len(self.completed_subtasks)]
        completed = chosen_color == current_target
        if completed:
            self.completed_subtasks.append(chosen_color)
        return chosen_color, completed


def rollout(model, tokenizer, environment, device, max_steps=4, verbose=True):
    """一个episode：从空历史开始，直到任务完成或达到步数上限。"""
    encoded = tokenizer(
        environment.instruction, truncation=True,
        max_length=32, return_tensors="pt",
    ).to(device)
    # 一组候选[4, 3, 16, 16] → 带批次维度[1, 4, 3, 16, 16]。
    images = environment.candidate_images.unsqueeze(0).to(device)
    model.eval()

    for step_number in range(1, max_steps + 1):
        if environment.is_done():
            break
        history_before = environment.completed_subtasks.copy()
        history_ids = torch.tensor(
            [environment.history_id()], dtype=torch.long, device=device,
        )  # [1]：本次只有一条样本，不是预先给好的第二步历史。
        with torch.no_grad():
            scores = model(
                candidate_images=images, history_ids=history_ids, **encoded,
            )  # [1, 4]：当前历史下，四个候选的分数。
            chosen_index = scores.argmax(dim=1).item()

        chosen_color, completed = environment.step(chosen_index)
        if verbose:
            result = "完成当前子任务" if completed else "未完成当前子任务"
            print(
                f"第{step_number}步 | 执行前历史={history_before} | "
                f"选择位置={chosen_index}（{chosen_color}） | 验证={result} | "
                f"执行后历史={environment.completed_subtasks}"
            )

    success = environment.is_done()
    if verbose:
        if success:
            print("STOP（规则）：两个子任务均已完成。")
        else:
            print("达到步数上限：任务未完成，不能报告成功。")
    return success


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = (
        Path(__file__).resolve().parent.parent
        / "models" / "best_history_candidate_scorer.pth"
    )
    if not model_path.is_file():
        raise FileNotFoundError("请先运行test47，生成历史融合模型的参数文件。")
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    model = HistoryCandidateScorer(model_name).to(device)
    model.load_state_dict(torch.load(
        model_path, map_location=device, weights_only=True,
    ))

    # 沿用第47课的测试场景，不挑选模型恰好答对的场景。
    # 一对样本提供场景与任务真值；执行中的历史从空开始、由实际结果更新。
    dataset = HistoryNavigationDataset(scenario_count=36, seed=13)
    scenario_count = len(dataset) // 2
    successful_episodes = 0
    print("设备：", device)
    print("仅推理，不重新训练。前两个场景展示完整执行过程。")
    for scenario_index in range(scenario_count):
        first_index = scenario_index * 2
        environment = ToyNavigationEnvironment(
            dataset.samples[first_index], dataset.samples[first_index + 1],
        )
        verbose = scenario_index < 2
        if verbose:
            print(f"\n场景{scenario_index + 1}：", environment.instruction)
            print("候选顺序：", environment.candidate_names)
        success = rollout(model, tokenizer, environment, device, verbose=verbose)
        successful_episodes += int(success)

    print(f"\n完整任务成功：{successful_episodes}/{scenario_count}")
    print(f"任务成功率：{successful_episodes / scenario_count:.2%}")
    print("单步准确率与完整任务成功率不是同一个指标。")
    print("观测和历史都不变时，确定性模型可能重复错误，步数上限防止死循环。")
    print("本课的完成验证和STOP均是已知规则，不代表已复现论文的验证模块。")


if __name__ == "__main__":
    main()
