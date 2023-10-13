import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import make_blobs

# 定义数据
X, y = make_blobs(n_samples = 10000, n_features = 3, centers = [[3, 3, 3], [0, 0, 0], [1, 1, 1], [2, 2, 2]],
                  cluster_std = [0.2, 0.1, 0.2, 0.2], random_state = 9)
 
# 降维到二维
pca = PCA(n_components = 2)
pca.fit(X)
'''
pca = PCA(n_components = 0.95)
当n_components=0.95表示指定了主成分累加起来至少占95%的那些成分
pca = PCA(n_components = 'mle')
当n_components='mle'表示让MLE算法自己选择降维维度的效果
'''

# 输出特征值
print(pca.explained_variance_)

# 输出特征向量
print(pca.components_)

# 降维后的数据
X_new = pca.transform(X)
print(X_new)
fig = plt.figure()
plt.scatter(X_new[:, 0], X_new[:, 1], marker = 'o')
plt.show()
