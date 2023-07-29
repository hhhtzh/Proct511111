from keplar.preoperator.preoperator import PreOperator


class ClusterJudge(PreOperator):
    def __init__(self,data):
        super().__init__()
        self.data = data

    def pre_do(self):
        dataSets=self.data.get_np_ds()
        if dataSets.shape[1] - 1 == 1:
            clusters = [1]
        elif dataSets.shape[1] - 1 == 2:
            clusters = [2, 4]
        else:
            clusters = [1, 2, 4, 8, 16]

        return clusters
        
    