# -*- coding: utf-8 -*-
"""
===================================
Demo of DBSCAN clustering algorithm
===================================

Finds core samples of high density and expands clusters from them.

"""

import numpy as np
import pandas as pd

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

'''
# Generate sample data
# --------------------
centers = [[1, 1,1], [-1, -1,-1], [1, -1,1]]
X, labels_true = make_blobs(
    n_samples=750, centers=centers, cluster_std=0.4, random_state=0
)
# 标准化
X = StandardScaler().fit_transform(X)
'''

# %%
# for fileNum in [72,73,74,75,76,77,78,79,80,81]:++
df = pd.read_csv("/home/hebaihe/tzh/All_Kepler/K1/Kepler/keplar/draw/name.csv")
# print(df)
df1 = df.iloc[:, -1]

for fileName in df1:
    # for fileNum in range(13,23):
    print("fileName = ", fileName)
    for epsilon in [0.2, 0.5, 0.8, 1, 1.5, 2, 2.5, 3, 4, 5, 10, 100]:
        X = np.loadtxt(r"/home/hebaihe/tzh/All_Kepler/K1/Kepler/datasets/pmlb/pmlb_txt/" + str(fileName) + ".txt", dtype=np.float64, skiprows=1)
        # X, Y = np.split(X_Y, (-1,), axis=1)
        # X = StandardScaler().fit_transform(X) #标准化
        # Compute DBSCAN
        # --------------
        db = DBSCAN(eps=epsilon, min_samples=2 * X.shape[1]).fit(X)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_  # 记录了每个数据点的分类结果，根据分类结果通过np.where就能直接取出对应类的所有数据索引了
        Data_res = np.where(db.labels_ == 1.0)[0]  # 保留所有分类是第二类（第一类是1.0）的数据集序号
        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        # 聚类过多或过少则舍掉
        if (n_clusters_ < 1 or n_clusters_ < 12): continue
        n_noise_ = list(labels).count(-1)
        # 半数以上的数据点没用上则舍弃
        # if (1.0*n_noise_/X.shape[0]>0.5): continue
        print("epsilon: ", epsilon)
        print("Estimated number of clusters: %d" % n_clusters_)
        print("Estimated number of noise points and totel points: %d " % n_noise_, " VS %d" % X.shape[0])
        '''
        print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
        print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
        print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
        print("Adjusted Rand Index: %0.3f" % metrics.adjusted_rand_score(labels_true, labels))
        print(
            "Adjusted Mutual Information: %0.3f"
            % metrics.adjusted_mutual_info_score(labels_true, labels)
        )
        print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, labels))

        '''

        # %%
        # Plot result
        # -----------
        import matplotlib.pyplot as plt

        # Black removed and is used for noise instead.
        unique_labels = set(labels)
        colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
        fig = plt.figure()
        # 创建3d绘图区域
        ax = plt.axes(projection='3d')

        for k, col in zip(unique_labels, colors):
            if k == -1:
                # Black used for noise.
                col = [0, 0, 0, 1]

            class_member_mask = labels == k

            xy = X[class_member_mask & core_samples_mask]
            # plt.plot(
            ax.plot3D(
                xy[:, 0],
                xy[:, 1],
                xy[:, 2],
                "o",
                markerfacecolor=tuple(col),
                markeredgecolor="k",
                markersize=14,
            )

            xy = X[class_member_mask & ~core_samples_mask]
            # plt.plot(
            ax.plot3D(
                xy[:, 0],
                xy[:, 1],
                xy[:, 2],
                "o",
                markerfacecolor=tuple(col),
                markeredgecolor="k",
                markersize=6,
            )

        plt.title("Estimated number of clusters: %d" % n_clusters_)
        figName = "cluster_stander_eps" + str(epsilon) + "_" + str(fileName) + ".jpg"
        plt.savefig('/home/hebaihe/tzh/All_Kepler/K1/Kepler/IMG_COLOR/LOG\\' + figName, dpi=1000,
                    bbox_inches='tight')
        plt.show()
