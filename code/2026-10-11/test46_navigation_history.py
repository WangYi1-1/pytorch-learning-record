"""最小导航例子：相同指令和候选视角，历史不同，下一步也不同。"""

import sys

import torch


sys.stdout.reconfigure(encoding="utf-8")


# 暂时把长指令拆成两个已知子任务；后续再研究自动拆分。
instruction = "先到红色区域，再到蓝色区域"
subtasks = ["red", "blue"]

# 两次决策都看到同样四个候选视角。
candidate_names = ["green", "red", "yellow", "blue"]

# 用候选图片的颜色编号模拟视觉特征；第0维表示候选视角。
color_to_id = {"red": 0, "green": 1, "blue": 2, "yellow": 3}
candidate_color_ids = torch.tensor(
    [color_to_id[name] for name in candidate_names],
    dtype=torch.long,
)


def choose_next_action(visited_subtasks):
    """根据已完成的子任务，选择当前目标对应的候选视角。"""
    step_index = len(visited_subtasks)

    if step_index >= len(subtasks):
        return "STOP", None

    target_name = subtasks[step_index]
    target_id = color_to_id[target_name]

    # 每张候选图与当前目标比较：匹配为1，否则为0。
    candidate_scores = (
        candidate_color_ids == target_id
    ).to(torch.float32)
    chosen_index = candidate_scores.argmax().item()

    return target_name, chosen_index


print("指令：", instruction)
print("候选视角：", candidate_names)
print("候选颜色编号：", candidate_color_ids.tolist())

for history in [[], ["red"], ["red", "blue"]]:
    current_target, action = choose_next_action(history)
    print(
        f"历史={history} | "
        f"当前目标={current_target} | "
        f"下一步动作={action}"
    )
