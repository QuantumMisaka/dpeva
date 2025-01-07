import numpy as np

from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal


# 尝试不同的 M 值
# M_values = range(5, 40)  # 尝试 M 从 1 到 20
# bic_values = []  # 保存 BIC 值
# X = all_desc_stru

# for M in M_values:
#     gmm = GaussianMixture(n_components=M, 
#                           covariance_type='full', 
#                           random_state=42,
#                           max_iter=100)
#     gmm.fit(X)
#     bic_values.append(gmm.bic(X))  # 计算 BIC 值

# # 选择使 BIC 最小的 M
# optimal_M = M_values[np.argmin(bic_values)]
# print(f"最优的 M 值: {optimal_M}")

# M = 27  # 高斯分布的数量
# gmm = GaussianMixture(n_components=M, covariance_type='full', max_iter=200, random_state=42)
# gmm.fit(X)  # 拟合GMM模型

