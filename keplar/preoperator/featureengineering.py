from keplar.preoperator.preoperator import PreOperator


class FeatureEngineering(PreOperator):
    def __init__(self):
        super().__init__()

    def do(self, data):
        raise NotImplemented