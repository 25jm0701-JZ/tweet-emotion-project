# 推文情绪分类：传统机器学习 vs RoBERTa 迁移学习

数据挖掘课程项目。在 `tweet_emotions.csv`（40,000 条推文、13 类情绪）上对比两条路线：
TF-IDF + 传统分类器，以及预训练 RoBERTa 的嵌入特征与全参数微调。

## 目录

```
tweet_emotion_project/
├── code/
│   ├── Main.py              入口，串起两个实验
│   ├── Experiment1.py       Experiment1：TF-IDF + NB / LR+SMOTE / XGBoost
│   └── Experiment2.py       Experiment2：RoBERTa 嵌入 + SVM/LR，以及微调
├── data/
│   └── tweet_emotions.csv   40,000 × 3，字段 id / sentiment / content
├── notebooks/
│   ├── 01_eda_tfidf.ipynb   探索版：清洗 + TF-IDF 的 IDF 权重分析
│   ├── 02_classifier_draft.ipynb  探索版：NB / SMOTE+LR 的原型
│   └── 03_colab_kaggle复现.ipynb   云端（Colab / Kaggle）复现 RoBERTa 实验
├── result/                  10 张图（其中 2 张微调图为 2026-11-07 重跑补回）
├── verify/                  素材库数字的核对脚本、指标与修订稿（2026-11-07 新增）
├── README_原版.txt          原始 README（未改动）
└── requirements.txt
```

`notebooks/` 里前两个文件是写论文时的草稿，逻辑已经全部并入 `code/`；保留是为了留下分析过程（例如 IDF 权重区间的筛选）。

## 实验设计

**Experiment 1（`EmotionClassifier`，传统方法）**
1. 正则清洗：去 `@` 符号、非字母字符、转小写，滤掉 `empty` 类和过短文本（注：代码里的 `@[\\w]*` 只删除 `@`，用户名会保留）；
2. TF-IDF：`max_features=10000`、`ngram_range=(1,2)`、英文停用词；
3. 多分类：MultinomialNB 基线 → 调 `alpha`（拉普拉斯平滑）→ SMOTE 过采样 + 逻辑回归；
4. 二分类（正/负）：逻辑回归 vs XGBoost。

**Experiment 2（`roberta_EmotionClassifier`，继承 Experiment1）**
1. `cardiffnlp/twitter-roberta-base-emotion` 冻结，取隐状态作为嵌入；
2. 在嵌入上训练轻量 Linear SVM 与逻辑回归，并画 t-SNE 看类别可分性；
3. 解冻全部参数用 Hugging Face `Trainer` 做全参数微调，输出混淆矩阵与 ROC。

## 运行

```bash
pip install -r requirements.txt
python code/Main.py
```

从项目根目录运行，结果写入 `result/`。Experiment 2 需要联网下载模型权重，有 CUDA 会快很多，纯 CPU 微调 4 万条会很慢。

## 与原始提交的差异

只改了两处路径，其余代码未动：
- `code/Main.py`：数据集路径 `'tweet_emotions.csv'` → `'data/tweet_emotions.csv'`（配合新目录结构）；
- `code/Experiment2.py:298`：ROC 图原本存到当前目录，改为 `result/finetune_roc_curve.png`，与 README 描述一致。

## 已知缺口

- 原 README 列了 10 张输出图。提交时 `result/` 只有 8 张，缺 `finetune_confusion_matrix.png` 和 `finetune_roc_curve.png`；这两张已于 2026-11-07 用 `verify/verify_experiment2.py` 在 RTX 3050 上重跑补回（重跑值与原提交可能相差 0.2pp 左右）。
- 附带的两个模型目录（`cardiffnlp/twitter-roberta-base-emotion-finetuned-emotion`、`twitter-roberta-base-emotion-finetuned-emotion`）是空的，权重本来就没拷进来；`Experiment2` 是从 Hugging Face 在线拉 `cardiffnlp/twitter-roberta-base-emotion`，不依赖这两个目录。
- 依赖未锁版本：`Trainer(tokenizer=...)` 在 transformers 5.x 已被移除，直接 `python code/Main.py` 会报 `TypeError`。复现请用 `verify/` 下的脚本，或先装 `transformers>=4.46,<5`。
- `Experiment1.py:76` 的清洗正则 `@[\\w]*` 只删掉了 `@` 符号，用户名仍然保留（详见 `verify/核对报告.md` 3.4）。

## 复现与核对

`verify/` 下是针对素材库数字的复现脚本与核对记录，不修改原始 `code/`：

- `python verify/verify_experiment1.py`：传统链路，约 30 秒，指标写入 `verify/metrics_exp1.json`；
- `python verify/verify_experiment2.py`：RoBERTa 冻结特征 + 微调，需要 GPU（4GB 显存即可，batch 32 + 梯度累积 2），约 20 分钟；
- 核对结论与修订文案：`verify/核对报告.md`、`verify/素材库_推特段落_修订版.docx`。

