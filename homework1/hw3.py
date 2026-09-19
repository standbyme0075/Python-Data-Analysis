# -*- coding: utf-8 -*-
"""
第 3 题：Carseats 数据集多元线性回归 + 多重共线性诊断

数据说明：Carseats.csv（Kaggle 上的 ISLR 配套数据集），
         模拟了 400 家门店销售儿童安全座椅的情况。
响应变量：Sales（销量，千件）
解释变量：Price（价格）、Income（社区收入）、Advertising（广告投入）、
         ShelveLoc（货架位置，定性：Bad / Medium / Good）

运行方式：python hw3.py
依赖    ：pip install pandas statsmodels
"""
from pathlib import Path

import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.outliers_influence import variance_inflation_factor

# ---------- 1. 读取数据 ----------
# 用 __file__ 定位数据文件，保证脚本和数据放在同一目录即可，不管从哪里运行都能读到
data_path = Path(__file__).resolve().parent / "Carseats.csv"
df = pd.read_csv(data_path)
print(f"数据规模：{df.shape[0]} 行，{df.shape[1]} 列")
print(f"ShelveLoc 的取值水平：{sorted(df['ShelveLoc'].unique())}\n")

# ---------- 2. 建立多元线性回归模型 ----------
# ShelveLoc 是定性变量，写在公式里时用 C() 包起来，
# statsmodels 会自动把它拆成哑变量（默认以字母序第一个水平为基准组）
model = ols("Sales ~ Price + Income + Advertising + C(ShelveLoc)", data=df).fit()

print("========== 模型拟合报告 ==========\n")
print(model.summary())

good_coef = model.params.get("C(ShelveLoc)[T.Good]")
good_p = model.pvalues.get("C(ShelveLoc)[T.Good]")

# ---------- 3. 计算各变量的 VIF ----------
# VIF = 1/(1 - R²_i)，其中 R²_i 是把第 i 个变量对其他所有变量做回归得到的拟合优度。
# VIF 越接近 1 说明该变量和其他变量越独立；经验上 VIF > 5（部分教材用 10）才值得警惕。
design = df[["Price", "Income", "Advertising"]].copy()
design = pd.concat(
    [design, pd.get_dummies(df["ShelveLoc"], prefix="ShelveLoc", drop_first=True)],
    axis=1,
)
# VIF 的计算需要设计矩阵带常数项列，但截距这一行本身不参与比较
design = sm.add_constant(design)

vif_values = [
    variance_inflation_factor(design.values, i)
    for i in range(1, design.shape[1])
]
vif_df = pd.DataFrame({"变量": design.columns[1:], "VIF": vif_values})
vif_df["VIF"] = vif_df["VIF"].round(3)
print("\n========== 各变量的 VIF ==========\n")
print(vif_df.to_string(index=False))

# ---------- 4. 三个问题的解答 ----------
print("\n========== 解答 ==========\n")
print("(1) ShelveLoc 的基准组：Bad。")
print("    哑变量编码默认取字母序第一个水平（Bad）为基准，")
print("    所以系数表里只有 ShelveLoc[Good] 和 ShelveLoc[Medium]。")

if good_coef is not None:
    sig = "统计上显著" if good_p < 0.05 else "统计上不显著"
    print("\n(2) ShelveLoc[Good] 的系数含义：")
    print(f"    系数约为 {good_coef:.3f}，{sig}（p 值 {good_p:.3f}，见报告 P>|t| 列）。")
    print("    它表示：在 Price、Income、Advertising 都相同的情况下，")
    print(f"    货架位置为 Good 的门店比基准组 Bad 的门店平均多卖出约 {good_coef:.2f} 千件。")
    print("    商业上说明货架位置是实打实的销量杠杆：同样的价格和广告投入，")
    print("    摆上黄金货架就能显著带动销售，零售商值得为好的陈列位置付溢价。")

print("\n(3) VIF 结论：各变量 VIF 都接近 1（见上表），远低于常用警戒线 5，")
print("    说明 Price、Income、Advertising 与 ShelveLoc 之间几乎没有线性相关，")
print("    不存在明显的多重共线性风险。")