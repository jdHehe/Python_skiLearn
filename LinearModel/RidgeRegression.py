# -*- coding:utf-8 -*-
print (__doc__)

import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

# 生成一个10*10的矩阵
X = 1. /(np.arange(1, 11) + np.arange(0, 10)[:, np.newaxis])


# 生成 10*1的向量
y = np.ones(10)


n_alphas = 200
#默认以自然对数为底
alphas = np.logspace(-10, -2, n_alphas)
print alphas

coefs = []
for a in alphas:
    ridge = linear_model.Ridge(alpha=a, fit_intercept=False)
    ridge.fit(X, y)
    coefs.append(ridge.coef_)

ax = plt.gca()

ax.plot(alphas, coefs)
ax.set_xscale('log')
ax.set_xlim(ax.get_xlim()[::-1])
plt.xlabel('alpha')
plt.ylabel('weights')
plt.title('Ridge coefficients as a function of the regularization')
plt.axis('tight')
plt.show()

