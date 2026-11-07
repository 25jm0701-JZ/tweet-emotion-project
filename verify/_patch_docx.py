# -*- coding: utf-8 -*-
"""把 素材库.docx 里 Twitter 项目那一段替换成核对后的版本（先备份）。"""
import shutil
from datetime import date

import docx
from docx.shared import Pt

SRC = r"C:\Users\-jmmmm\Desktop\U\中介申请\素材库.docx"
BAK = SRC.replace(".docx", f"_备份_{date.today():%Y-%m-%d}.docx")

# (style, [(text, bold), ...])
ITEMS = [
    ("Normal", [("时间：2025.09 – 2025.12（按实际课程学期填写）", False)]),
    ("Heading 3", [("项目简介", False)]),
    ("Normal", [
        ("基于公开推文数据集（tweet_emotions.csv，40,000 条、13 个标签，清洗后 39,170 条 / 12 类情绪），"
         "构建「12 类多分类 + 正负二分类」两条情绪识别流程。数据本身高噪声（@ 符号、表情符、拼写随意）"
         "且类别极不均衡（anger 仅 110 条，neutral 8,635 条），项目重点放在清洗、类别不均衡处理与预训练模型迁移三条线上，"
         "用于量化「传统 TF-IDF 基线」与「RoBERTa 迁移」在同一任务上的差距。", False)]),
    ("Heading 3", [("职责", False)]),
    ("List Bullet", [
        ("数据清洗与特征工程：", True),
        (" 基于公开推文数据集（tweet_emotions.csv，40,000 条、13 个标签），用正则去除 @ 符号与非字母字符、转小写，"
         "剔除 empty 与长度≤1 的文本，得到 39,170 条、12 类情绪；完成标签映射与类别分布检查"
         "（neutral 8,635、worry 8,459、happiness 5,209、anger 110）。TF-IDF 取 10,000 维、1–2 gram、英文停用词。", False)]),
    ("List Bullet", [
        ("算法选型与基线构建：", True),
        (" MultinomialNB 基线准确率 0.317、Macro-F1 0.129；调 α=0.5 后 0.320 / 0.139。"
         "二分类对比逻辑回归（0.746）与 XGBoost（0.728），验证高维稀疏特征下线性模型更优。", False)]),
    ("List Bullet", [
        ("类别不均衡处理：", True),
        (" 仅在训练集上做 SMOTE（31,336 → 82,896 样本）。LR+SMOTE 把长尾类召回拉起来："
         "boredom 0.00→0.22、fun 0.01→0.20、hate 0.08→0.32、love 0.36→0.41，Macro-F1 由 0.175 升至 0.204；"
         "代价是整体准确率由 0.352 降至 0.269，且测试集中仅 22 条的 anger 仍为 0。"
         "因此把它定位为「以整体准确率换长尾召回」的方案，而非全面更优。", False)]),
    ("List Bullet", [
        ("预训练大模型微调与降维分析：", True),
        (" 以 cardiffnlp/twitter-roberta-base-emotion 冻结 [CLS] 提取 768 维特征训练 LinearSVC，准确率 0.794；"
         "再进行全参数微调（3 epoch、lr 2e-5、batch 64、max_len 128），准确率 0.8053、Weighted F1 0.8053、AUC 0.8771。"
         "用 t-SNE 对 500 条测试样本的 768 维特征降维，观察正负情绪的聚类分离。", False)]),
    ("Heading 3", [("业绩", False)]),
    ("List Bullet", [
        ("多分类结果：", True),
        (" MNB 0.317 / 0.129，调 α 后 0.320 / 0.139；LR+SMOTE Macro-F1 0.204（准确率 0.269）。", False)]),
    ("List Bullet", [
        ("二分类结果：", True),
        (" TF-IDF 逻辑回归 0.746 > XGBoost 0.728；RoBERTa 冻结特征 + LinearSVC 0.794；"
         "全参数微调 0.8053（Weighted F1 0.8053、AUC 0.8771）。80% 这一档只适用于二分类任务，12 类任务仍在 0.27–0.35。", False)]),
    ("List Bullet", [
        ("可解释性输出：", True),
        (" 二分类 LR 的高权重词为 love +4.62、happy +4.00、amazing +2.93，sad −5.06、hate −4.12、sucks −2.85"
         "（notebook 原始运行 C=1.0 时约 +3.73 / −3.63），说明模型主要依赖直白情绪词，符合任务预期。", False)]),
    ("List Bullet", [
        ("方法论沉淀：", True),
        (" 跑通「清洗 → TF-IDF 基线 → 不均衡处理 → 预训练模型微调 → 误差分析」的完整链路，并保留可复现脚本与逐项指标。", False)]),
    ("Heading 3", [("简历表述版", False)]),
    ("Normal", [
        ("海量非结构化数据清洗与预处理：", True),
        (" 对 4 万条英文推文做正则清洗（去 @ 符号、非字母字符、转小写）与标签过滤，剔除 empty 与过短文本，"
         "得到 39,170 条 / 12 类数据，并完成分布检查。", False)]),
    ("Normal", [
        ("应对极端不均衡的数据痛点（方案优化）：", True),
        (" 在训练集上引入 SMOTE（31k → 83k），把 boredom / fun / hate / love 的召回由 0.00 / 0.01 / 0.08 / 0.36 "
         "提升到 0.22 / 0.20 / 0.32 / 0.41，Macro-F1 由 0.175 提升到 0.204；同时如实记录整体准确率由 0.352 降至 0.269 的代价。", False)]),
    ("Normal", [
        ("传统机器学习与深度预训练模型的双轨评估：", True),
        (" TF-IDF + MNB / LR / XGBoost 建立基线（二分类 0.746 vs 0.728），对比 RoBERTa 冻结特征 + LinearSVC（0.794）"
         "与全参数微调（0.8053，Weighted F1 0.8053、AUC 0.8771）。", False)]),
    ("Normal", [
        ("模型可解释性与可视化输出：", True),
        (" 用 t-SNE 把 768 维文本特征降到二维；提取 LR 高权重词（love +4.62、happy +4.00、amazing +2.93；"
         "sad −5.06、hate −4.12、sucks −2.85），说明模型的判断依据。", False)]),
]


def main():
    shutil.copy2(SRC, BAK)
    d = docx.Document(SRC)
    paras = d.paragraphs

    start = end = None
    for i, p in enumerate(paras):
        t = p.text.strip()
        if start is None and t.startswith("海外社媒舆情挖掘与用户洞察调研"):
            start = i
        elif start is not None and p.style.name == "Heading 2":
            end = i
            break
    assert start is not None and end is not None and end > start, (start, end)
    print(f"替换范围：段落 {start + 1} – {end - 1}（标题「{paras[start].text.strip()}」保留）")

    # 先记录要删除的 XML 元素
    doomed = [paras[i]._p for i in range(start + 1, end)]
    anchor = paras[end]._p

    # 插入新段落（先 append 再移动到锚点前，保持顺序）
    for style, runs in ITEMS:
        p = d.add_paragraph(style=style)
        for text, bold in runs:
            r = p.add_run(text)
            r.bold = bold
            r.font.size = Pt(10.5)
        anchor.addprevious(p._p)

    for el in doomed:
        el.getparent().remove(el)

    d.save(SRC)
    print("已保存：", SRC)
    print("备份：", BAK)

    d2 = docx.Document(SRC)
    hit = [p.text for p in d2.paragraphs if p.text.strip().startswith("海外社媒")]
    print("校验：", hit, "段落总数", len(d2.paragraphs))


if __name__ == "__main__":
    main()
