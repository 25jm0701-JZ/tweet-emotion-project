# -*- coding: utf-8 -*-
"""用 verify/preds.npz 生成 t-SNE、混淆矩阵、ROC 三张图（在带 matplotlib 的环境下运行）。"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "result_repro")
os.makedirs(OUT, exist_ok=True)
d = np.load(os.path.join(HERE, "preds.npz"))
print("keys:", list(d.keys()))

xy, lab = d["tsne_xy"], d["tsne_labels"]
plt.figure(figsize=(10, 8))
sns.scatterplot(x=xy[:, 0], y=xy[:, 1], hue=lab, palette="deep", s=60)
plt.title("t-SNE Visualization of Sentiment Vectors")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "tsne_visualization.png"), dpi=120)
plt.close()

yte, preds, probs = d["yte"], d["preds"], d["probs"]
cm = confusion_matrix(yte, preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
plt.xlabel("Predicted"); plt.ylabel("True")
plt.title("Fine-Tuned Model Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "finetune_confusion_matrix.png"), dpi=120)
plt.close()

fpr, tpr, _ = roc_curve(yte, probs)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC)")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "finetune_roc_curve.png"), dpi=120)
plt.close()
print("saved to", OUT, "AUC=", round(roc_auc, 4))
