# -*- coding: utf-8 -*-
# 第 3 题：Carseats 数据集多元线性回归 + VIF 诊断
from pathlib import Path

import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.outliers_influence import variance_inflation_factor

# 读取数据（与脚本同目录）
df = pd.read_csv(Path(__file__).resolve().parent / "Carseats.csv")

# ShelveLoc 是定性变量，公式里用 C() 表示
model = ols("Sales ~ Price + Income + Advertising + C(ShelveLoc)", data=df).fit()
print(model.summary())

# 计算 VIF（构造含截距的设计矩阵，截距行不参加比较）
design = pd.concat(
    [
        df[["Price", "Income", "Advertising"]],
        pd.get_dummies(df["ShelveLoc"], prefix="ShelveLoc", drop_first=True),
    ],
    axis=1,
)
design = sm.add_constant(design)
vif_df = pd.DataFrame(
    {
        "变量": design.columns[1:],
        "VIF": [
            variance_inflation_factor(design.values, i)
            for i in range(1, design.shape[1])
        ],
    }
)
print("\nVIF：")
print(vif_df.round(3).to_string(index=False))

# 解答
good_coef = model.params["C(ShelveLoc)[T.Good]"]
print("\n基准组：Bad（按字母序取第一个水平）")
print(f"ShelveLoc[Good] 系数 = {good_coef:.2f}：价格、收入、广告相同时，Good 货架比 Bad 货架平均多卖 {good_coef:.2f} 千件")
print("VIF 都接近 1，小于 5，无明显多重共线性。")