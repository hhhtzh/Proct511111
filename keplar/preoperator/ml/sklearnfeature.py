
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold

from keplar.preoperator.featureengineering import FeatureEngineering


class SklearnFeature(FeatureEngineering):
    def __init__(self, method):
        super().__init__()
        self.method = method

    def do(self, data):
        if "PCA" in self.method:
            p = PCA()
            pd = data.pd_data
            data.pd_data = p.fit_transform(pd)
        if "filter" in self.method:
            var = VarianceThreshold(threshold=1)
            pd = data.pd_data
            data.pd_data = var.fit_transform(pd)
        return data
