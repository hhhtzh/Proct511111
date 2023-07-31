from keplar.posoperator.posoperator import PosOperator


class Pruning(PosOperator):
    def __init__(self,programs,fit_list,op_list,n_cluster):
        super().__init__()
        self.n_cluster = n_cluster
        self.op_list = op_list
        self.fit_list = fit_list
        self.programs = programs

    def pos_do(self):
        pass
