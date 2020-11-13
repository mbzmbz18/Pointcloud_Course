# 文件功能： 实现 K-Means 算法

import random
import numpy as np
import matplotlib.pyplot as plt



class K_Means(object):
    # k是分组数； tolerance‘中心点误差’； max_iter是迭代次数
    def __init__(self, n_clusters=2, tolerance=0.0001, max_iter=300):
        self.k_ = n_clusters
        self.tolerance_ = tolerance
        self.max_iter_ = max_iter

    def distance(self, a, b):
        difference = (a-b)**2

    def fit(self, data):
        # 作业1
        # 屏蔽开始

        # Choose random points as centroid
        n_point = data.shape[0]
        indices_samples = random.sample(range(n_point), self.k_)
        self.centers = data[indices_samples]
        old_centers = np.copy(self.centers)
        cluster_label = np.zeros(n_point)

        # Outer Iteration
        for it in range(self.max_iter_):

            print(it)
            # Calculate distance to centroid
            # Iterate over all data points
            for i in range(n_point):
                current_point = data[i]
                current_distance = np.zeros(self.k_)
                # Number of old_center = k
                for k in range(len(old_centers)):
                    current_centroid = old_centers[k]
                    current_distance[k] = np.linalg.norm(current_centroid - current_point)
                current_label = np.argmin(current_distance)
                cluster_label[i] = current_label

            for k in range(self.k_):
                # Find corresponding data which belonging to label k
                points = data[cluster_label==k]
                center = np.mean(points, axis=0)
                self.centers[k, :] = center

            tolerance = np.sum(np.abs(self.centers - old_centers))
            if tolerance < self.k_ * self.tolerance_:
                break
            old_centers = np.copy(self.centers)


        # 屏蔽结束

    def predict(self, p_datas):
        result = []
        # 作业2

        for point in p_datas:
            distance = np.linalg.norm(self.centers - point, axis=1)
            result.append(np.argmin(distance))

        # 屏蔽结束
        return result


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
    fig = plt.figure(figsize=(10, 8))
    #plt.axis([-10, 15, -5, 15])
    ax1 = fig.add_subplot(111)
    ax1.scatter(X1[:, 0], X1[:, 1], s=5)
    ax1.scatter(X2[:, 0], X2[:, 1], s=5)
    ax1.scatter(X3[:, 0], X3[:, 1], s=5)
    plt.show()
    return X

#######################################################################################
# Main Function
#x = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])
true_Mu = [[0.5, 0.5], [5.5, 2.5], [1, 7]]
true_Var = [[2, 3], [2, 1], [2, 2]]
x = generate_X(true_Mu, true_Var)
k_means = K_Means(n_clusters=3)
k_means.fit(x)

cat = k_means.predict(x)
#print(cat)

