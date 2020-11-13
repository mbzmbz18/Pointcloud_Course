# 文件功能：实现 GMM 算法

import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
#plt.style.use('seaborn')


class GMM(object):
    def __init__(self, n_clusters, max_iter=50):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.data = None
        self.Mu = None
        self.Var = None
        self.W = None
        self.Pi = None
        self.loglh = []

    # Function for initialization at beginning
    def GMM_init(self, data):
        self.data = data
        n_point = len(data)
        indices_samples = random.sample(range(n_point), self.n_clusters)
        centroid = data[indices_samples]
        # Initialize useful parameters
        self.Mu = centroid
        self.Var = self.Var = np.ones((self.n_clusters, 2))
        self.Pi = np.asarray([1 / self.n_clusters] * self.n_clusters)
        self.W = np.ones((n_point, self.n_clusters)) / self.n_clusters


    # Compute the log-likelihood
    def logLH(self):
        pdfs = np.zeros(((len(self.data), self.n_clusters)))
        for i in range(self.n_clusters):
            pdfs[:, i] = self.Pi[i] * multivariate_normal.pdf(self.data, self.Mu[i], self.Var[i])
        log_LH = np.mean(np.log(np.sum(pdfs, axis=1)))
        return log_LH

    # Clustering main function
    def fit(self, data):
        # 作业3
        # 屏蔽开始
        self.GMM_init(data)
        self.loglh.append(self.logLH())
        # Outer iteration
        for iter in range(self.max_iter):
            # E-step
            pdfs = np.zeros(((len(data), self.n_clusters)))
            for i in range(self.n_clusters):
                pdfs[:, i] = self.Pi[i] * multivariate_normal.pdf(self.data, self.Mu[i], self.Var[i])
            self.W = pdfs / np.sum(pdfs, axis=1).reshape(-1, 1)
            # M-step
            self.Pi = np.sum(self.W, axis=0) / self.data.shape[0]
            self.Mu = np.zeros((self.n_clusters, 2))
            for i in range(self.n_clusters):
                self.Mu[i] = np.average(self.data, axis=0, weights=self.W[:, i])
            self.Var = np.zeros((self.n_clusters, 2))
            for i in range(self.n_clusters):
                self.Var[i] = np.average((self.data - self.Mu[i]) ** 2, axis=0, weights=self.W[:, i])
            self.loglh.append(self.logLH())
            if abs(self.loglh[-1] - self.loglh[-2]) < 0.00000001:
                print(iter)
                break

            # 屏蔽结束
    
    def predict(self, data):
        # 屏蔽开始
        result = []
        pdfs = np.zeros((data.shape[0], self.n_clusters))
        for i in range(self.n_clusters):
            pdfs[:, i] = self.Pi[i] * multivariate_normal.pdf(data, self.Mu[i], np.diag(self.Var[i]))
        W = pdfs / pdfs.sum(axis=1).reshape(-1, 1)
        result = np.argmax(W, axis=1)
        return result

        # 屏蔽结束


# 生成仿真数据
def generate_X(true_Mu, true_Var):
    # 第一簇的数据
    num1, mu1, var1 = 400, true_Mu[0], true_Var[0]
    X1 = np.random.multivariate_normal(mu1, np.diag(var1), num1)
    # 第二簇的数据
    num2, mu2, var2 = 600, true_Mu[1], true_Var[1]
    X2 = np.random.multivariate_normal(mu2, np.diag(var2), num2)
    # 第三簇的数据
    num3, mu3, var3 = 1000, true_Mu[2], true_Var[2]
    X3 = np.random.multivariate_normal(mu3, np.diag(var3), num3)
    # 合并在一起
    X = np.vstack((X1, X2, X3))
    # 显示数据
    plt.figure(figsize=(10, 8))
    plt.axis([-10, 15, -5, 15])
    plt.scatter(X1[:, 0], X1[:, 1], s=5)
    plt.scatter(X2[:, 0], X2[:, 1], s=5)
    plt.scatter(X3[:, 0], X3[:, 1], s=5)
    #plt.show()
    return X

##########################################################
# Main Function
# 生成数据
true_Mu = [[0.5, 0.5], [5.5, 2.5], [1, 7]]
true_Var = [[1, 3], [2, 2], [6, 2]]
X = generate_X(true_Mu, true_Var)

gmm = GMM(n_clusters=3)
gmm.fit(X)
cat = gmm.predict(X)
#print(cat)
# 初始化

