# -*- coding: utf-8 -*-
"""
复现 Experiment1（传统机器学习链路），逐项核对素材库中的数字。

与原版 code/Experiment1.py 完全一致的设置：
  清洗正则 / test_size=0.2 / random_state=42 / stratify / TF-IDF(max_features=10000,
  ngram_range=(1,2), stop_words='english') / MNB / alpha 搜索 / SMOTE+LR(C=0.5)
  二分类：LR(class_weight='balanced', C=0.5) 与 XGBoost(n_estimators=200, depth=6, lr=0.1)

额外补充（原版代码没有打印、素材库却引用了的指标）：
  - SMOTE+LR 的 classification_report（原版只画混淆矩阵）
  - 不做 SMOTE 的 LR 各类召回，用来检验“召回率从 0 提升”这一说法
  - 二分类 LR 的 Top 词系数，用来核对“+0.87 / -0.96”这一说法
"""
import json
import os
import re
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             f1_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import xgboost as xgb

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "metrics_exp1.json")
report_lines = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    report_lines.append(s)


def per_class(y_true, y_pred, names):
    return {n: round(float(recall_score(y_true, y_pred, labels=[i], average=None)[0]), 4)
            for i, n in enumerate(names)}


def summarize(y_true, y_pred, names, tag):
    acc = accuracy_score(y_true, y_pred)
    macro = f1_score(y_true, y_pred, average="macro")
    weighted = f1_score(y_true, y_pred, average="weighted")
    log(f"[{tag}] accuracy={acc:.4f} macroF1={macro:.4f} weightedF1={weighted:.4f}")
    log(classification_report(y_true, y_pred, target_names=names, zero_division=0))
    return {
        "accuracy": round(float(acc), 4),
        "macro_f1": round(float(macro), 4),
        "weighted_f1": round(float(weighted), 4),
        "per_class_recall": per_class(y_true, y_pred, names),
    }


def main():
    t0 = time.time()
    df = pd.read_csv(os.path.join(ROOT, "data", "tweet_emotions.csv"))
    log(f"原始数据: {df.shape[0]} 行, sentiment 取值 {df['sentiment'].nunique()} 个")
    log(df["sentiment"].value_counts().to_string())

    # --- 与原版一致的清洗 ---
    df["cleaned_text"] = df["content"].apply(lambda x: re.sub(r"@[\\w]*", "", str(x)))
    df["cleaned_text"] = df["cleaned_text"].apply(lambda x: re.sub(r"[^a-zA-Z']", " ", x))
    df["cleaned_text"] = df["cleaned_text"].str.lower().str.strip()
    cleaned = df[df["sentiment"] != "empty"].copy()
    cleaned = cleaned[cleaned["cleaned_text"].str.len() > 1].reset_index(drop=True)
    log(f"清洗前 {len(df)} 行 -> 清洗后 {len(cleaned)} 行; 清洗后类别数 {cleaned['sentiment'].nunique()}")

    le = LabelEncoder()
    X = cleaned["cleaned_text"]
    y = le.fit_transform(cleaned["sentiment"])
    names = le.classes_

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    vec = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), stop_words="english")
    Xtr = vec.fit_transform(X_train)
    Xte = vec.transform(X_test)
    log(f"TF-IDF 维度: {Xtr.shape} (训练), {Xte.shape} (测试), 词表 {len(vec.vocabulary_)}")

    res = {"data": {
        "raw_rows": int(df.shape[0]),
        "raw_labels": int(df["sentiment"].nunique()),
        "cleaned_rows": int(len(cleaned)),
        "cleaned_labels": int(cleaned["sentiment"].nunique()),
        "train": int(len(X_train)), "test": int(len(X_test)),
        "tfidf_features": int(Xtr.shape[1]),
    }}

    # --- MNB baseline ---
    mnb = MultinomialNB().fit(Xtr, y_train)
    res["mnb_default"] = summarize(y_test, mnb.predict(Xte), names, "MNB default")

    # --- alpha 搜索 ---
    alphas, sweep = [0.01, 0.1, 0.5, 1.0, 2.0], {}
    for a in alphas:
        m = MultinomialNB(alpha=a).fit(Xtr, y_train)
        p = m.predict(Xte)
        sweep[str(a)] = round(float(accuracy_score(y_test, p)), 4)
        log(f"alpha={a}: acc={sweep[str(a)]:.4f}")
    res["mnb_alpha_sweep_acc"] = sweep

    m05 = MultinomialNB(alpha=0.5).fit(Xtr, y_train)
    res["mnb_alpha_0.5"] = summarize(y_test, m05.predict(Xte), names, "MNB alpha=0.5")

    # --- LR 不做 SMOTE（用于核对“召回率从 0 提升”） ---
    lr_raw = LogisticRegression(max_iter=2000, C=0.5, random_state=42).fit(Xtr, y_train)
    res["lr_no_smote"] = summarize(y_test, lr_raw.predict(Xte), names, "LR (no SMOTE)")

    # --- SMOTE + LR ---
    Xs, ys = SMOTE(random_state=42, sampling_strategy="not majority", k_neighbors=5).fit_resample(Xtr, y_train)
    log(f"SMOTE 后训练集: {Xs.shape}")
    lr_sm = LogisticRegression(max_iter=2000, C=0.5, random_state=42).fit(Xs, ys)
    res["lr_smote"] = summarize(y_test, lr_sm.predict(Xte), names, "LR + SMOTE")
    res["lr_smote"]["train_after_smote"] = int(Xs.shape[0])

    # --- 二分类 ---
    neg = ["worry", "sadness", "hate", "boredom", "anger"]
    pos = [e for e in names if e not in neg + ["neutral"]]
    dfb = cleaned[cleaned["sentiment"] != "neutral"].copy()
    dfb["binary_label"] = dfb["sentiment"].apply(lambda x: 1 if x in pos else 0)
    Xb_tr, Xb_te, yb_tr, yb_te = train_test_split(
        dfb["cleaned_text"], dfb["binary_label"], test_size=0.2,
        random_state=42, stratify=dfb["binary_label"])
    vecb = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    Xbtr = vecb.fit_transform(Xb_tr)
    Xbte = vecb.transform(Xb_te)
    log(f"二分类样本: train={len(Xb_tr)}, test={len(Xb_te)}, 正类占比={yb_te.mean():.4f}")

    lrb = LogisticRegression(class_weight="balanced", max_iter=2000, C=0.5,
                             random_state=42).fit(Xbtr, yb_tr)
    pb = lrb.predict(Xbte)
    log(f"[Binary LR] acc={accuracy_score(yb_te, pb):.4f} f1={f1_score(yb_te, pb):.4f}")
    log(classification_report(yb_te, pb, target_names=["Negative", "Positive"], zero_division=0))
    res["binary_lr"] = {
        "accuracy": round(float(accuracy_score(yb_te, pb)), 4),
        "f1": round(float(f1_score(yb_te, pb)), 4),
        "report": classification_report(yb_te, pb, target_names=["Negative", "Positive"],
                                        zero_division=0, output_dict=True),
    }

    xg = xgb.XGBClassifier(objective="binary:logistic", n_estimators=200, max_depth=6,
                           learning_rate=0.1, random_state=42, verbosity=0).fit(Xbtr, yb_tr)
    px = xg.predict(Xbte)
    log(f"[Binary XGBoost] acc={accuracy_score(yb_te, px):.4f} f1={f1_score(yb_te, px):.4f}")
    res["binary_xgb"] = {
        "accuracy": round(float(accuracy_score(yb_te, px)), 4),
        "f1": round(float(f1_score(yb_te, px)), 4),
    }

    # --- 二分类 LR 词系数（核对 +0.87 / -0.96） ---
    inv = {i: w for w, i in vecb.vocabulary_.items()}
    coef = lrb.coef_[0]
    order = np.argsort(coef)
    top_pos = [(inv[i], round(float(coef[i]), 4)) for i in order[::-1][:10]]
    top_neg = [(inv[i], round(float(coef[i]), 4)) for i in order[:10]]
    log("Top+ :", top_pos)
    log("Top- :", top_neg)
    res["binary_top_positive"] = top_pos
    res["binary_top_negative"] = top_neg

    # --- 提及清洗的真实效果 ---
    res["mention_cleaning"] = {
        "tweets_with_at_raw": int(df["content"].astype(str).str.contains("@").sum()),
        "note": "正则 r'@[\\\\w]*' 只删除 '@' 本身，用户名保留",
    }

    res["elapsed_sec"] = round(time.time() - t0, 1)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    with open(os.path.join(os.path.dirname(OUT), "report_exp1.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    log(f"耗时 {res['elapsed_sec']}s -> {OUT}")


if __name__ == "__main__":
    main()
