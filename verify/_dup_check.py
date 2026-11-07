# -*- coding: utf-8 -*-
import pandas as pd
import re
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/tweet_emotions.csv")
df = df[df["sentiment"] != "empty"].copy()
df["cleaned_text"] = df["content"].apply(lambda x: re.sub(r"@[\\w]*", "", str(x)))
df["cleaned_text"] = df["cleaned_text"].apply(lambda x: re.sub(r"[^a-zA-Z']", " ", x)).str.lower().str.strip()
df = df[df["cleaned_text"].str.len() > 1].reset_index(drop=True)
dup = df["cleaned_text"].duplicated(keep=False)
print("total rows", len(df), "rows in duplicate groups", int(dup.sum()))
print("unique texts", df["cleaned_text"].nunique())
tr, te = train_test_split(df, test_size=0.2, random_state=42, stratify=df["sentiment"])
overlap = set(tr["cleaned_text"]) & set(te["cleaned_text"])
print("test rows whose cleaned text also appears in train:",
      int(te["cleaned_text"].isin(overlap).sum()), "of", len(te))
print("tweets containing @:", int(df["content"].astype(str).str.contains("@").sum()))
print(df.loc[df["content"].astype(str).str.contains("@"), ["content", "cleaned_text"]].head(4).to_string())
