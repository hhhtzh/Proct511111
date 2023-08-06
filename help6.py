import numpy as np
from sklearn.cluster import DBSCAN

from keplar.data.data import Data

data = Data("pmlb", "1027_ESL", ["x1", "x2", "x3", 'y'])
data.read_file()
dataSets = data.get_np_x()
db = DBSCAN(eps=1, min_samples=2 * dataSets.shape[1]).fit(dataSets)
labels = db.labels_  # 记录了每个数据点的分类结果，根据分类结果通过np.where就能直接取出对应类的所有数据索引了
db_sum = []
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
for j in range(n_clusters_):
    temp_db = []
    for i in range(len(labels)):
        if labels[i] == j:
            temp_db.append(dataSets[i])
    temp_db = np.array(temp_db)
    print(j)
    print("--------")
    print(temp_db)
    db_sum.append(temp_db)
n_noise_ = list(labels).count(-1)
print(f"划分子块{n_clusters_}个")
print(f"统计噪声数据共{n_noise_}条")
if n_clusters_ < 1:
    print("划分数据集失败")
