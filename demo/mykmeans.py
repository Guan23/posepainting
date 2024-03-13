#!usr/bin/env python
# _*_ encoding: utf-8 _*_
# Author: Guan
# Create Time: 2023/3/13 下午10:23

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
from scipy import stats


def my_kmeans():
    X = np.array(
        [
            [6.36502629e+03, 1.92219391e+03, 8.57622429e+00],
            [1.10756284e+04, 3.36465005e+03, 1.42884880e+01],
            [1.08757014e+04, 3.33613348e+03, 1.41992395e+01],
            [6.59266155e+03, 1.97744686e+03, 8.64006653e+00],
            [6.46155885e+03, 1.92923836e+03, 8.46805044e+00],
            [6.45718750e+03, 1.94186360e+03, 8.50540171e+00],
        ]
    )
    scaler = MinMaxScaler(feature_range=(0,1))  # 实例化
    scaler = scaler.fit(X)  # fit，在这里本质是生成min(x)和max(x)
    result = scaler.transform(X)  # 通过接口导出结果
    print(result)

    y_pred = DBSCAN(eps=0.5, min_samples=2).fit_predict(result)
    # y_pred = KMeans(n_clusters=2).fit_predict(X)  # 下标
    numb = stats.mode(y_pred)[0][0]  # 众数
    print(numb)
    print(y_pred)
    print(X[y_pred == numb][:, 2])
    print(np.mean(X[y_pred == numb][:, 2]))


def my_knn():
    # 定义一个数组
    X = np.array([[-1, -1],
                  [-2, -1],
                  [-3, -2],
                  [1, 1],
                  [2, 1],
                  [3, 2]])

    nbrs = NearestNeighbors(n_neighbors=3, algorithm="ball_tree").fit(X)
    # 返回距离每个点k个最近的点和距离指数，indices可以理解为表示点的下标，distances为距离
    distances, indices = nbrs.kneighbors(X)
    print(distances, "\n\n", indices)
    print(indices[:, 1:] - 1)


if __name__ == "__main__":
    print("\n--------------------- start ---------------------\n")
    my_kmeans()
    print("\n---------------------- end ----------------------\n")
