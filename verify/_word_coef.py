# -*- coding: utf-8 -*-
"""查几个词在二分类 LR 中的系数与排名，用于核对素材库的 +0.87 / -0.96 说法。"""
import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/tweet_emotions.csv")
df["cleaned_text"] = df["content"].apply(lambda x: re.sub(r"@[\\w]*", "", str(x)))
df["cleaned_text"] = df["cleaned_text"].apply(lambda x: re.sub(r"[^a-zA-Z']", " ", x)).str.lower().str.strip()
df = df[df["sentiment"] != "empty"]
df = df[df["cleaned_text"].str.len() > 1]
neg = ["worry", "sadness", "hate", "boredom", "anger"]
df = df[df["sentiment"] != "neutral"].copy()
df["y"] = df["sentiment"].apply(lambda x: 0 if x in neg else 1)
tr, te, ytr, yte = train_test_split(df["cleaned_text"], df["y"], test_size=0.2,
                                    random_state=42, stratify=df["y"])
v = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
Xtr = v.fit_transform(tr)
Xte = v.transform(te)
m = LogisticRegression(class_weight="balanced", max_iter=2000, C=0.5, random_state=42).fit(Xtr, ytr)
coef = m.coef_[0]
inv = {i: w for w, i in v.vocabulary_.items()}
order = np.argsort(coef)[::-1]
rank = {inv[i]: r + 1 for r, i in enumerate(order)}
for w in ["amazing", "happy", "sucks", "stupid", "love", "sad"]:
    idx = v.vocabulary_.get(w)
    print(f"{w:9s} coef={coef[idx]:+.4f} rank={rank[w]}/10000" if idx is not None else f"{w}: not in vocab")
print("max coef", coef.max().round(4), "min coef", coef.min().round(4))
