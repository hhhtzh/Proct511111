import numpy as np
from sklearn.cluster import DBSCAN

from keplar.preoperator.preoperator import PreOperator


class SklearnDBscan(PreOperator):
    def __init__(self):
        super().__init__()

    def do(self,data):
        epsilon = 1e-5
        dataSets=data.get_np_ds()
        db = DBSCAN(eps=epsilon, min_samples=2 * dataSets.shape[1]).fit(dataSets)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_  # 记录了每个数据点的分类结果，根据分类结果通过np.where就能直接取出对应类的所有数据索引了
        print(labels)
        # Number of clusters in labels, ignoring noise if present.