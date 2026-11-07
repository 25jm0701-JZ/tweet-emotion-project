# -*- coding: utf-8 -*-
"""根据实测指标生成修订后的素材库段落（docx + md），数字自动填入。"""
import json
import os

from docx import Document
from docx.shared import Pt

HERE = os.path.dirname(os.path.abspath(__file__))
e1 = json.load(open(os.path.join(HERE, "metrics_exp1.json"), encoding="utf-8"))
e2 = json.load(open(os.path.join(HERE, "metrics_exp2.json"), encoding="utf-8"))

m, a = e1["mnb_default"], e1["mnb_alpha_0.5"]
lr0, sm = e1["lr_no_smote"], e1["lr_smote"]
bl, xg = e1["binary_lr"], e1["binary_xgb"]
fz, ft = e2["frozen_linear_svc"], e2["finetune"]
R0, R = lr0["per_class_recall"], sm["per_class_recall"]

f = lambda x: f"{x:.3f}"
p = lambda x: f"{x:.1%}"

paragraphs = []


def H2(t):
    paragraphs.append(("h2", t))


def H3(t):
    paragraphs.append(("h3", t))


def P(t):
    paragraphs.append(("p", t))


def B(t):
    paragraphs.append(("b", t))


H2("海外社媒舆情挖掘与用户洞察调研（X / Twitter 市场分析）")
P("时间：2025.09 – 2025.12（请按实际学期确认；原文写 2026.12 落在未来）")

H3("项目简介")
P("基于公开推文数据集（tweet_emotions.csv，40,000 条、13 个标签，清洗后 39,170 条 / 12 类情绪），"
  "构建「12 类多分类 + 正负二分类」两条情绪识别流程。数据本身高噪声（@ 符号、表情符、拼写随意）"
  "且类别极不均衡（anger 仅 110 条，neutral 8,635 条），项目重点放在清洗、类别不均衡处理与预训练模型迁移三条线上，"
  "用于量化「传统 TF-IDF 基线」与「RoBERTa 迁移」在同一任务上的差距。")

H3("职责")
B(f"数据清洗与特征工程：用正则去除 @ 符号与非字母字符、转小写，剔除 empty 与长度≤1 的文本，"
  f"得到 39,170 条 / 12 类；完成标签映射与类别分布检查（neutral 8,635、worry 8,459、happiness 5,209、anger 110）。"
  f"TF-IDF 取 10,000 维、1–2 gram、英文停用词。")
B(f"算法选型与基线构建：MultinomialNB 基线准确率 {f(m['accuracy'])}、Macro-F1 {f(m['macro_f1'])}；"
  f"调 α=0.5 后 {f(a['accuracy'])} / {f(a['macro_f1'])}。二分类用逻辑回归（{f(bl['accuracy'])}）"
  f"与 XGBoost（{f(xg['accuracy'])}）对比，验证高维稀疏特征下线性模型更优。")
B(f"类别不均衡处理：仅在训练集上做 SMOTE（31,336 → {sm['train_after_smote']:,} 样本）。"
  f"LR+SMOTE 把长尾类召回拉起来（boredom {R0['boredom']:.2f}→{R['boredom']:.2f}、"
  f"fun {R0['fun']:.2f}→{R['fun']:.2f}、hate {R0['hate']:.2f}→{R['hate']:.2f}、"
  f"love {R0['love']:.2f}→{R['love']:.2f}），Macro-F1 由 {f(lr0['macro_f1'])} 升至 {f(sm['macro_f1'])}；"
  f"代价是整体准确率由 {f(lr0['accuracy'])} 降至 {f(sm['accuracy'])}，且测试集中仅 22 条的 anger 仍为 0。"
  f"因此把 SMOTE 定位为「以整体准确率换长尾召回」的方案，而非全面更优。")
B(f"预训练模型迁移与降维分析：以 cardiffnlp/twitter-roberta-base-emotion 冻结 [CLS] 768 维特征训练 LinearSVC，"
  f"准确率 {f(fz['accuracy'])}；再进行全参数微调（3 epoch、lr 2e-5、batch 64、max_len 128），"
  f"准确率 {f(ft['accuracy'])}、Weighted F1 {f(ft['weighted_f1'])}、AUC {f(ft['auc'])}。"
  f"用 t-SNE 对 500 条测试样本的 768 维特征降维，观察正负情绪的聚类分离。")

H3("业绩")
B(f"多分类结果：MNB {f(m['accuracy'])} / {f(m['macro_f1'])}，调 α 后 {f(a['accuracy'])} / {f(a['macro_f1'])}；"
  f"LR+SMOTE Macro-F1 {f(sm['macro_f1'])}（准确率 {f(sm['accuracy'])}）。")
B(f"二分类结果：TF-IDF 逻辑回归 {f(bl['accuracy'])} > XGBoost {f(xg['accuracy'])}；"
  f"RoBERTa 冻结特征 + LinearSVC {f(fz['accuracy'])}；全参数微调 {f(ft['accuracy'])}"
  f"（Weighted F1 {f(ft['weighted_f1'])}、AUC {f(ft['auc'])}）。"
  f"80% 这一档只适用于二分类任务，12 类任务仍在 0.27–0.35。")
B("可解释性输出：二分类 LR 的高权重词为 love +4.62、happy +4.00、amazing +2.93，"
  "sad −5.06、hate −4.12、sucks −2.85（notebook 原始运行 C=1.0 时约 +3.73 / −3.63），"
  "说明模型主要依赖直白情绪词，符合任务预期。")
B("方法论沉淀：跑通「清洗 → TF-IDF 基线 → 不均衡处理 → 预训练模型微调 → 误差分析」的完整链路，"
  "并保留可复现脚本与逐项指标。")

H3("简历表述版")
B("海量非结构化数据清洗：对 4 万条英文推文做正则清洗（去 @ 符号、非字母字符、转小写）与标签过滤，"
  "剔除 empty 与过短文本，得到 39,170 条 / 12 类数据，并完成分布检查。")
B(f"应对极端不均衡（方案优化）：在训练集上引入 SMOTE（31k → 83k），把 boredom / fun / hate / love 的召回"
  f"由 {R0['boredom']:.2f} / {R0['fun']:.2f} / {R0['hate']:.2f} / {R0['love']:.2f} "
  f"提升到 {R['boredom']:.2f} / {R['fun']:.2f} / {R['hate']:.2f} / {R['love']:.2f}，"
  f"Macro-F1 由 {f(lr0['macro_f1'])} 提升到 {f(sm['macro_f1'])}；"
  f"同时如实记录整体准确率由 {f(lr0['accuracy'])} 降至 {f(sm['accuracy'])} 的代价，避免只报收益。")
B(f"双轨评估：TF-IDF + MNB / LR / XGBoost 建立基线（二分类 {f(bl['accuracy'])} vs {f(xg['accuracy'])}），"
  f"对比 RoBERTa 冻结特征 + LinearSVC（{f(fz['accuracy'])}）与全参数微调"
  f"（{f(ft['accuracy'])}，Weighted F1 {f(ft['weighted_f1'])}、AUC {f(ft['auc'])}）。")
B("可解释性与可视化：用 t-SNE 对 768 维文本特征降维到二维；提取 LR 高权重词"
  "（love +4.62、happy +4.00、amazing +2.93；sad −5.06、hate −4.12、sucks −2.85），说明模型的判断依据。")

doc = Document()
style = doc.styles["Normal"]
style.font.name = "Microsoft YaHei"
style.font.size = Pt(11)
for kind, text in paragraphs:
    if kind == "h2":
        doc.add_heading(text, level=2)
    elif kind == "h3":
        doc.add_heading(text, level=3)
    elif kind == "b":
        doc.add_paragraph(text, style="List Bullet")
    else:
        doc.add_paragraph(text)

out_docx = os.path.join(HERE, "素材库_推特段落_修订版.docx")
doc.save(out_docx)

# 同步一份 markdown，便于 diff
md = ["# 素材库 · Twitter 项目段落（修订版，数字已填）", ""]
for kind, text in paragraphs:
    if kind == "h2":
        md.append(f"## {text}")
    elif kind == "h3":
        md.append(f"### {text}")
    elif kind == "b":
        md.append(f"- {text}")
    else:
        md.append(text)
    md.append("")
out_md = os.path.join(HERE, "素材库_推特段落_修订版_数字版.md")
with open(out_md, "w", encoding="utf-8") as fh:
    fh.write("\n".join(md))
print("written", out_docx)
print("finetune", ft["accuracy"], ft["weighted_f1"], ft["auc"])
