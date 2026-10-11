# 第46—48课：历史状态与连续任务

- test46_navigation_history.py：用几种预设历史演示规则决策，不训练模型。
- test47_train_with_history.py：训练 BERT + CNN + 历史 Embedding 的候选打分模型。
- test48_navigation_rollout.py：导入第47课结构，加载其参数，再执行两步玩具任务。

截至2026-10-11：第47课已逐段学完，第48课只讲到环境初始化；代码已运行不等于课程已学完。

## 运行条件

已有 Python、PyTorch 与 transformers 环境；小型预训练 BERT
google/bert_uncased_L-2_H-128_A-2 的模型和分词器必须已缓存。
代码使用 local_files_only=True，不会自动下载预训练文件。

从仓库根目录依次运行：

```powershell
python code/2026-10-11/test47_train_with_history.py
python code/2026-10-11/test48_navigation_rollout.py
```

先运行第47课训练，才能运行第48课。参数保存在这些脚本所在目录上一层的
models/best_history_candidate_scorer.pth；在本仓库布局中即 code/models/。
第48课只加载参数，不训练、不覆盖参数文件。

模型参数、缓存与数据不随本次记录上传。环境中的完成检查与 STOP 是规则，
不是学出来的模块；合成颜色任务结果不能当作真实视觉语言导航结果。
