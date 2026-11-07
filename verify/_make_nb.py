# -*- coding: utf-8 -*-
"""生成 Colab / Kaggle 复现 notebook。"""
import json
import os

CELLS = []


def md(src):
    CELLS.append({"cell_type": "markdown", "metadata": {},
                  "source": src.strip("\n").splitlines(keepends=True)})


def code(src):
    CELLS.append({"cell_type": "code", "execution_count": None, "metadata": {},
                  "outputs": [],
                  "source": src.strip("\n").splitlines(keepends=True)})


md(r"""
# tweet_emotion_project：RoBERTa 实验云端复现

本 notebook 用于在 **Google Colab** 或 **Kaggle** 上复跑 `tweet_emotion_project`，重点复现：

- Experiment 1：TF-IDF + MNB / SMOTE+LR / 二分类 LR、XGBoost（CPU 即可，约 1 分钟）
- Experiment 2：`cardiffnlp/twitter-roberta-base-emotion`
  - 冻结特征 + LinearSVC
  - 全参数微调 3 epoch + 混淆矩阵 + ROC
  - t-SNE 降维

## 平台设置

**Colab**
1. 菜单 `代码执行程序 → 更改运行时类型`
2. 硬件加速器选 `T4 GPU`，保存
3. 直接从上到下运行本 notebook

**Kaggle**
1. 右上角 `Settings → Accelerator` 选 `GPU T4 x2`
2. `Settings → Internet` 打开（需要下载 HF 模型）
3. 把项目 zip 作为 Dataset 上传，或直接在 Notebook 里上传

> 说明：原版 `code/Experiment2.py` 用了 `Trainer(tokenizer=...)`，该参数在 transformers 5.x 已被移除，所以下面固定安装 `transformers>=4.46,<5`。
""")

md("## 1. 检查 GPU")
code("""
!nvidia-smi
""")

md("## 2. 安装依赖（固定 transformers 4.x）")
code("""
%pip -q install "transformers>=4.46,<5" "datasets<4" "imbalanced-learn" "xgboost" "seaborn"
""")

md(r"""
## 3. 获取项目文件

**Colab**：先在本地把 `tweet_emotion_project` 整个文件夹压缩成 `tweet_emotion_project.zip`，运行下面的单元格选择上传。

**Kaggle**：把同一个 zip 作为 Dataset 上传，然后把 `ZIP` 改成 `/kaggle/input/.../tweet_emotion_project.zip`。
""")
code("""
import os, zipfile, sys

ZIP = "tweet_emotion_project.zip"          # Kaggle 改成 /kaggle/input/.../tweet_emotion_project.zip
if not os.path.exists(ZIP):
    from google.colab import files
    print("请选择 tweet_emotion_project.zip 上传")
    files.upload()

with zipfile.ZipFile(ZIP) as z:
    z.extractall(".")

# 兼容 zip 里多一层目录的情况
if os.path.isdir("tweet_emotion_project"):
    os.chdir("tweet_emotion_project")
print("cwd =", os.getcwd())
print(sorted(os.listdir(".")))
""")

md("## 4. 确认数据")
code("""
import pandas as pd
df = pd.read_csv("data/tweet_emotions.csv")
print(df.shape)
print(df["sentiment"].value_counts())
""")

md(r"""
## 5. 运行 Experiment 1（传统机器学习）

原版入口是 `code/Main.py`，但它会连着跑 Experiment 2。这里只跑 Experiment 1，方便先确认传统链路数字。
""")
code("""
import sys
sys.path.insert(0, "code")
from Experiment1 import EmotionClassifier

clf = EmotionClassifier("data/tweet_emotions.csv")
clf.run_Experiment1()
""")

md(r"""
## 6. 运行 Experiment 2（RoBERTa）

- 冻结模型抽 `[CLS]` 768 维特征 → LinearSVC / LR
- 解冻全参数微调 3 epoch（lr=2e-5，batch=64，max_len=128）
- 输出混淆矩阵、ROC、t-SNE

T4 上大约 8–12 分钟。
""")
code("""
from Experiment2 import roberta_EmotionClassifier

rob = roberta_EmotionClassifier(clf)
rob.run_Experiment2()
""")

md("## 7. 查看结果图")
code("""
from IPython.display import Image, display
for f in [
    "result/1_origin_distribution.png",
    "result/2_cm_multinomial_nb.png",
    "result/3_cm_nb_alpha_0.5.png",
    "result/4_cm_lr_smote.png",
    "result/5_cm_binary_lr.png",
    "result/6_cm_binary_xgboost.png",
    "result/traditional_models_comparison.png",
    "result/tsne_visualization.png",
    "result/finetune_confusion_matrix.png",
    "result/finetune_roc_curve.png",
]:
    if os.path.exists(f):
        display(Image(f))
    else:
        print("缺失:", f)
""")

md(r"""
## 8. 可选：用核对脚本复核素材库数字

`verify/verify_experiment1.py` 会额外打印 SMOTE+LR 的完整分类报告（原版代码只画混淆矩阵），
`verify/verify_experiment2.py` 会把指标写入 `verify/metrics_exp2.json`。

```bash
!python verify/verify_experiment1.py
!python verify/verify_experiment2.py
```

> 注意：`verify/` 只读取 `data/tweet_emotions.csv`，不会覆盖 `result/` 里的原始图片；
> 它把新图写到 `verify/result_repro/`。
""")

code("""
!python verify/verify_experiment1.py
""")

code("""
!python verify/verify_experiment2.py
""")

md(r"""
## 9. 常见问题

| 现象 | 原因 / 处理 |
|---|---|
| `TypeError: Trainer.__init__() got an unexpected keyword argument 'tokenizer'` | transformers 5.x 已移除该参数，重跑第 2 格安装 `<5` 的版本 |
| `KeyError: 'text'` 或 collator 报错 | 自行改写脚本时忘了 `remove_unused_columns=False` |
| 下载模型很慢 / 超时 | 设置 `os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"` 后再 import transformers |
| CUDA out of memory | 把 batch size 降到 32，或 `max_length` 降到 64 |
| 结果与素材库数字略有出入 | 冻结/微调结果受 GPU、随机种子、库版本影响，±1pp 属正常 |
""")

nb = {
    "cells": CELLS,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "T4"},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}
out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "notebooks", "03_colab_kaggle复现.ipynb")
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print("written", out, len(CELLS), "cells")
