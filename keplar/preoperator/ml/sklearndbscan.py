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