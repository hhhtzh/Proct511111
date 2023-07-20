import numpy as np
from sklearn.cluster import KMeans

from keplar.preoperator.preoperator import PreOperator


class SklearnKmeans(PreOperator):
    def __init__(self, n_clusters):
        super().__init__()
        self.subRegions = None
        self.n_clusters = n_clusters

    def do(self, data):
        global arr
        dataSets_x = data.get_np_x()
        dataSets = data.get_np_ds()
        db_sum = []
        k_means = KMeans(n_clusters=self.n_clusters, random_state=10).fit(dataSets_x)
        labels = k_means.labels_
        for cluster in range(0, self.n_clusters):
            dataIndex = np.where(labels == cluster * 1.0)[0]  # 保留所有分类是第一类（第二类是1.0）的数据集序号
            for i in range(0, dataIndex.shape[0]):  # 反推当前子块索引对应数据
                if i == 0:
                    arr = np.array([dataSets[dataIndex[0]]])  # 升维
                else:
                    arr = np.append(arr, np.array([dataSets[dataIndex[i]]]), axis=0)
            if cluster == 0:
                self.subRegions = [arr]  # 合并所有子块的数据
            else:
                self.subRegions.extend([arr])
        return self.subRegions, self.n_clusters
