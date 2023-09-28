import numpy as np
from matplotlib import pyplot as plt

from sklearn.manifold import TSNE


def TsneDraw(data_x,labels,epsilon):
    tsne = TSNE(n_components=2, random_state=42, perplexity=data_x.shape[0] - 15)
    # 使用t-SNE对数据进行降维
    embedded_data = tsne.fit_transform(data_x)
    unique_labels = np.unique(labels)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))

    plt.figure(figsize=(10, 8))

    for i, label in enumerate(unique_labels):
        mask = labels == label
        plt.scatter(embedded_data[mask, 0], embedded_data[mask, 1], c=colors[i],
                    label=f'Cluster {label}')

    plt.title('eps:' + str(epsilon) + ',t-SNE Visualization of DBSCAN Clusters')
    plt.legend()
    # plt.show()