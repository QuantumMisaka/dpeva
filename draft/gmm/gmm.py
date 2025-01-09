import numpy as np

from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal

# 载入描述符数据

# 尝试不同的 M 值
# M_values = range(5, 40)  # 尝试 M 从 5 到 40
# bic_values = []  # 保存 BIC 值

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

# 生成GMM模型
# gmm = GaussianMixture(n_components=optimal_M, covariance_type='full', max_iter=200, random_state=42)
# gmm.fit(X)  # 拟合GMM模型

