# -*- coding: utf-8 -*-
"""
复现 Experiment2（RoBERTa 冻结特征 + 全参数微调），核对素材库里的
79.40% / 80.76% / Weighted F1 0.81 / AUC 0.88。

与 code/Experiment2.py 一致的设置：
  - 二分类数据（去掉 neutral），test_size=0.2, random_state=42, stratify
  - 使用原始 content 文本（不是清洗后的 cleaned_text）
  - max_length=128, padding='max_length'
  - 冻结模型取 [CLS] hidden state -> LinearSVC(C=0.1) / LogisticRegression
  - t-SNE 取测试集前 500 条
  - 全参数微调：3 epoch, lr=2e-5, batch=64, weight_decay=0.01

仅有的工程性调整（不改变实验协议）：
  - 用 DataCollatorWithPadding 动态 padding 以加速 CPU 训练
  - 训练日志与结果写到 verify/ 下，不覆盖 result/ 原图

用法：
  python verify/verify_experiment2.py            # 完整复现
  set QUICK=1 && python verify/verify_experiment2.py   # 小样本冒烟测试
"""
import gc
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_PLOT = True
    try:
        import seaborn as sns
        HAVE_SNS = True
    except Exception:
        HAVE_SNS = False
except Exception:
    HAVE_PLOT = False
    HAVE_SNS = False
from transformers import (AutoModel, AutoModelForSequenceClassification,
                          AutoTokenizer, DataCollatorWithPadding, Trainer,
                          TrainingArguments)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, "result_repro")
os.makedirs(OUTDIR, exist_ok=True)

CKPT = os.environ.get("MODEL_PATH", "cardiffnlp/twitter-roberta-base-emotion")
QUICK = os.environ.get("QUICK") == "1"
EPOCHS = float(os.environ.get("EPOCHS", "3"))
MAXLEN = int(os.environ.get("MAXLEN", "128"))
BS = int(os.environ.get("BS", "64"))
GAS = int(os.environ.get("GAS", "1"))
torch.set_num_threads(os.cpu_count() or 8)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

lines = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    lines.append(s)


def per_class(y_true, y_pred, names):
    from sklearn.metrics import recall_score
    return {n: round(float(recall_score(y_true, y_pred, average=None)[i]), 4)
            for i, n in enumerate(names)}


def main():
    t0 = time.time()
    log(f"device={DEVICE}, threads={torch.get_num_threads()}, quick={QUICK}, epochs={EPOCHS}")

    df = pd.read_csv(os.path.join(ROOT, "data", "tweet_emotions.csv"))
    neg = ["worry", "sadness", "hate", "boredom", "anger", "neutral"]
    df = df[df["sentiment"] != "empty"]
    df = df[df["content"].astype(str).str.len() > 1]
    dfb = df[df["sentiment"] != "neutral"].copy()
    dfb["binary_label"] = dfb["sentiment"].apply(lambda x: 0 if x in neg else 1)
    train_df, test_df = train_test_split(dfb, test_size=0.2, random_state=42,
                                         stratify=dfb["binary_label"])
    if QUICK:
        train_df = train_df.sample(n=600, random_state=0)
        test_df = test_df.sample(n=300, random_state=0)
    log(f"二分类样本: train={len(train_df)}, test={len(test_df)}, "
        f"正类占比={test_df['binary_label'].mean():.4f}")

    tok = AutoTokenizer.from_pretrained(CKPT)
    log(f"tokenizer={tok.__class__.__name__}")

    def encode(texts):
        return tok(list(texts), padding="max_length", max_length=MAXLEN,
                   truncation=True, return_tensors="pt")

    # ---------- 1. 冻结特征 ----------
    model = AutoModel.from_pretrained(CKPT).to(DEVICE).eval()
    hidden = getattr(model.config, "hidden_size", None)
    log(f"hidden_size={hidden}")

    def extract(texts, bs=None):
        bs = bs or BS
        outs = []
        with torch.no_grad():
            for i in range(0, len(texts), bs):
                batch = encode(texts[i:i + bs])
                batch = {k: v.to(DEVICE) for k, v in batch.items()}
                outs.append(model(**batch).last_hidden_state[:, 0].cpu().numpy())
                if (i // bs) % 50 == 0:
                    log(f"  embedding {i + len(batch['input_ids'])}/{len(texts)}")
        return np.concatenate(outs)

    cache = os.path.join(HERE, "emb_cache.npz")
    if os.path.exists(cache) and not QUICK:
        z = np.load(cache)
        Xtr, Xte, ytr, yte = z["Xtr"], z["Xte"], z["ytr"], z["yte"]
        log(f"载入嵌入缓存 {Xtr.shape} / {Xte.shape}")
    else:
        Xtr = extract(list(train_df["content"]))
        Xte = extract(list(test_df["content"]))
        ytr = train_df["binary_label"].to_numpy()
        yte = test_df["binary_label"].to_numpy()
        if not QUICK:
            np.savez(cache, Xtr=Xtr, Xte=Xte, ytr=ytr, yte=yte)
    log(f"特征形状: {Xtr.shape} / {Xte.shape}")

    del model
    gc.collect()

    res = {"data": {"train": int(len(train_df)), "test": int(len(test_df)),
                    "hidden_size": hidden, "max_len": MAXLEN}}

    svm = LinearSVC(C=0.1, max_iter=10000, random_state=42, dual="auto").fit(Xtr, ytr)
    p_svm = svm.predict(Xte)
    res["frozen_linear_svc"] = {
        "accuracy": round(float(accuracy_score(yte, p_svm)), 4),
        "f1": round(float(f1_score(yte, p_svm)), 4),
        "per_class_recall": per_class(yte, p_svm, ["Negative", "Positive"]),
    }
    log(f"[冻结特征 + LinearSVC] acc={res['frozen_linear_svc']['accuracy']:.4f} "
        f"f1={res['frozen_linear_svc']['f1']:.4f}")

    lr = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42).fit(Xtr, ytr)
    p_lr = lr.predict(Xte)
    res["frozen_lr"] = {
        "accuracy": round(float(accuracy_score(yte, p_lr)), 4),
        "f1": round(float(f1_score(yte, p_lr)), 4),
    }
    log(f"[冻结特征 + LR] acc={res['frozen_lr']['accuracy']:.4f} f1={res['frozen_lr']['f1']:.4f}")

    # ---------- 2. t-SNE ----------
    n = min(len(Xte), 500)
    emb = TSNE(n_components=2, random_state=42).fit_transform(Xte[:n])
    np.savez(os.path.join(HERE, "preds.npz"), tsne_xy=emb, tsne_labels=yte[:n],
             yte=yte, ytr=ytr)
    if HAVE_PLOT:
        plt.figure(figsize=(10, 8))
        if HAVE_SNS:
            sns.scatterplot(x=emb[:, 0], y=emb[:, 1], hue=yte[:n], palette="deep", s=60)
        else:
            for lab, color in [(0, "tab:blue"), (1, "tab:orange")]:
                m = yte[:n] == lab
                plt.scatter(emb[m, 0], emb[m, 1], s=60, c=color, label=str(lab))
            plt.legend(title="label")
        plt.title("t-SNE Visualization of Sentiment Vectors")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTDIR, "tsne_visualization.png"), dpi=120)
        plt.close()
        log("t-SNE saved")
    else:
        log("t-SNE 坐标已保存（当前环境无 matplotlib，稍后单独绘图）")

    # 先落盘冻结特征部分的结果，避免微调失败时丢数据
    with open(os.path.join(HERE, "metrics_exp2_partial.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)

    # ---------- 3. 全参数微调 ----------
    class DS(torch.utils.data.Dataset):
        def __init__(self, texts, labels):
            self.texts, self.labels = list(texts), list(labels)

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, i):
            return {"text": self.texts[i], "labels": self.labels[i]}

    collator = DataCollatorWithPadding(tok, return_tensors="pt")

    def make_collator(features):
        batch = collator([tok(f["text"], truncation=True, max_length=MAXLEN)
                          for f in features])
        batch["labels"] = torch.tensor([f["labels"] for f in features])
        return batch

    ft = AutoModelForSequenceClassification.from_pretrained(
        CKPT, num_labels=2, ignore_mismatched_sizes=True).to(DEVICE)

    def metrics(pred):
        labels = pred.label_ids
        preds = np.argmax(pred.predictions, axis=1)
        return {"accuracy": accuracy_score(labels, preds),
                "f1": f1_score(labels, preds, average="weighted")}

    args = TrainingArguments(
        output_dir=os.path.join(HERE, "_trainer_out"),
        num_train_epochs=EPOCHS,
        learning_rate=2e-5,
        per_device_train_batch_size=BS,
        gradient_accumulation_steps=GAS,
        per_device_eval_batch_size=BS,
        weight_decay=0.01,
        eval_strategy="epoch",
        logging_steps=max(1, len(train_df) // BS // 5),
        report_to="none",
        save_strategy="no",
        disable_tqdm=False,
        seed=42,
        remove_unused_columns=False,
    )
    trainer = Trainer(model=ft, args=args, train_dataset=DS(train_df["content"], ytr),
                      eval_dataset=DS(test_df["content"], yte),
                      data_collator=make_collator, compute_metrics=metrics)
    t1 = time.time()
    trainer.train()
    res["finetune_seconds"] = round(time.time() - t1, 1)
    log(f"微调耗时 {res['finetune_seconds']}s")

    pred = trainer.predict(DS(test_df["content"], yte))
    logits = pred.predictions
    preds = np.argmax(logits, axis=1)
    probs = torch.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()

    acc = accuracy_score(yte, preds)
    wf1 = f1_score(yte, preds, average="weighted")
    auc = roc_auc_score(yte, probs)
    res["finetune"] = {
        "accuracy": round(float(acc), 4),
        "weighted_f1": round(float(wf1), 4),
        "auc": round(float(auc), 4),
        "per_class_recall": per_class(yte, preds, ["Negative", "Positive"]),
        "report": classification_report(yte, preds, target_names=["Negative", "Positive"],
                                        output_dict=True, zero_division=0),
    }
    log(f"[微调] acc={acc:.4f} weightedF1={wf1:.4f} AUC={auc:.4f}")
    log(classification_report(yte, preds, target_names=["Negative", "Positive"], zero_division=0))

    cm = confusion_matrix(yte, preds)
    fpr, tpr, _ = roc_curve(yte, probs)
    d = np.load(os.path.join(HERE, "preds.npz"))
    np.savez(os.path.join(HERE, "preds.npz"), tsne_xy=d["tsne_xy"],
             tsne_labels=d["tsne_labels"], yte=yte, ytr=ytr, preds=preds, probs=probs)

    if HAVE_PLOT:
        plt.figure(figsize=(6, 5))
        if HAVE_SNS:
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                        xticklabels=["Negative", "Positive"],
                        yticklabels=["Negative", "Positive"])
        else:
            plt.imshow(cm, cmap="Blues")
            for i in range(2):
                for j in range(2):
                    plt.text(j, i, str(cm[i, j]), ha="center", va="center")
            plt.xticks([0, 1], ["Negative", "Positive"])
            plt.yticks([0, 1], ["Negative", "Positive"])
        plt.xlabel("Predicted"); plt.ylabel("True")
        plt.title("Fine-Tuned Model Confusion Matrix")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTDIR, "finetune_confusion_matrix.png"), dpi=120)
        plt.close()

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {auc:.2f})")
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic (ROC)")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTDIR, "finetune_roc_curve.png"), dpi=120)
        plt.close()
        log("混淆矩阵与 ROC 已保存")
    else:
        log("混淆矩阵/ROC 数据已保存（稍后单独绘图）")

    res["elapsed_sec"] = round(time.time() - t0, 1)
    with open(os.path.join(HERE, "metrics_exp2.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    with open(os.path.join(HERE, "report_exp2.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()

