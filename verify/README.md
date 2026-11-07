# verify/ 说明

这里是对《素材库.docx》中 Twitter 项目段落做核对时留下的脚本、指标与产物。**不修改 `code/` 原始提交**，所有新结果写在本目录。

## 文件

| 文件 | 作用 |
|---|---|
| `核对报告.md` | 逐条核对结论（22 条，11 ✅ / 7 ⚠️ / 4 ❌）与证据 |
| `素材库_推特段落_修订版.docx` / `.md` | 改好的素材库段落，可直接替换原文 |
| `verify_experiment1.py` | 复现传统链路，含原代码漏掉的 SMOTE+LR 分类报告、无 SMOTE 对照 |
| `verify_experiment2.py` | 复现 RoBERTa 冻结特征 + LinearSVC + t-SNE + 全参数微调 |
| `metrics_exp1.json` / `metrics_exp2.json` | 两次复现的全部指标 |
| `report_exp1.txt` / `report_exp2.txt` | 运行日志 |
| `result_repro/` | 重跑产生的 t-SNE、微调混淆矩阵、ROC |
| `make_plots.py` | 从 `preds.npz` 单独画图（环境里没有 matplotlib 时用） |
| `download_model.py` | 从 hf-mirror 并行下载 RoBERTa 权重 |

## 复现方法

```bash
# 1) 依赖（Python 3.11）。torch 建议装 CUDA 版，CPU 微调要 2 小时以上
pip install scikit-learn imbalanced-learn xgboost pandas numpy \
            transformers "datasets<4" torch seaborn matplotlib

# 2) 模型权重（约 500MB；国内直连 HuggingFace 会超时，用镜像）
set HF_ENDPOINT=https://hf-mirror.com
python verify/download_model.py          # 下载到 verify/_model

# 3) 传统链路，约 30 秒
python verify/verify_experiment1.py

# 4) RoBERTa：4GB 显存用 batch 32 + 梯度累积 2（等效 batch 64）
set MODEL_PATH=verify\_model
set BS=32
set GAS=2
python verify/verify_experiment2.py
```

## 本次复现环境

| 项 | 值 |
|---|---|
| 系统 | Windows 11，16 逻辑核，RTX 3050 Laptop 4GB |
| Python | 3.11 |
| torch | 2.11.0+cu128 |
| transformers | 5.18.0（`verify_experiment2.py` 已适配 v5） |
| scikit-learn | 1.9.0 |

## 结果速查

| 指标 | 本次重跑 | 素材库 |
|---|---|---|
| MNB accuracy / Macro-F1 | 0.3171 / 0.1293 | 0.32 / 0.13 |
| α=0.5 Macro-F1 | 0.1394 | 0.14 |
| 二分类 LR / XGBoost | 0.7464 / 0.7282 | 0.75 / 0.73 |
| 冻结特征 + LinearSVC | 0.7940 | 79.40% |
| 全参数微调 | 0.8053（F1 0.8053，AUC 0.8771） | 80.76% / 0.81 / 0.88 |
| LR+SMOTE accuracy / Macro-F1 | 0.2691 / 0.2038 | 未提 |

> 模型权重（`verify/_model`，500MB）与嵌入缓存没有随项目保留；重跑时按上面的第 2 步重新下载即可。
