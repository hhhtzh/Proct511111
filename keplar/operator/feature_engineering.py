from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold

from keplar.operator.operator import Operator


class FeatureEngineering(Operator):
    def __init__(self, method, data):
        super().__init__()
        self.data = data
        self.method = method

    def do(self, population=None):
        if "PCA" in self.method:
            p = PCA()
            pd = self.data.pd_data
            self.data.pd_data=p.fit_transform(pd)
        if "filter" in self.method:
            var = VarianceThreshold(threshold=1)
            pd=self.data.pd_data
            self.data.pd_data=var.fit_transform(pd)


