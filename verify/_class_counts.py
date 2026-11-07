# -*- coding: utf-8 -*-
import pandas as pd
import re

df = pd.read_csv("data/tweet_emotions.csv")
df["t"] = df["content"].apply(lambda x: re.sub(r"@[\\w]*", "", str(x)))
df["t"] = df["t"].apply(lambda x: re.sub(r"[^a-zA-Z']", " ", x)).str.lower().str.strip()
raw = df["sentiment"].value_counts()
cl = df[df["sentiment"] != "empty"]
cl = cl[cl["t"].str.len() > 1]
clean = cl["sentiment"].value_counts()
out = pd.DataFrame({"raw": raw, "cleaned": clean}).fillna(0).astype(int)
print(out.to_string())
print("rows", len(df), "->", len(cl))
print("classes raw", df["sentiment"].nunique(), "cleaned", cl["sentiment"].nunique())
