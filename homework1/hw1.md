# 第 1 题：一元线性回归的两个残差性质

模型为 $y_i=\beta_0+\beta_1x_i+e_i$，OLS 最小化残差平方和：

$$
RSS(\beta_0,\beta_1)=\sum_{i=1}^{n}(y_i-\beta_0-\beta_1x_i)^2
$$

最小值处 RSS 对 $\beta_0$、$\beta_1$ 的偏导均为 0。

对 $\beta_0$ 求偏导并令其为 0：

$$
\frac{\partial RSS}{\partial \beta_0}=-2\sum_{i=1}^{n}(y_i-\hat\beta_0-\hat\beta_1x_i)=0
$$

其中 $y_i-\hat\beta_0-\hat\beta_1x_i=e_i$，所以 $\sum_{i=1}^{n}e_i=0$。

对 $\beta_1$ 求偏导并令其为 0：

$$
\frac{\partial RSS}{\partial \beta_1}=-2\sum_{i=1}^{n}x_i(y_i-\hat\beta_0-\hat\beta_1x_i)=0
$$

所以 $\sum_{i=1}^{n}x_ie_i=0$。

这两个等式合起来就是正规方程。注意：两个性质成立的前提是模型含截距，若强制直线过原点则不成立。