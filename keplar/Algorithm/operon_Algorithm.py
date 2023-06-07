from keplar.Algorithm.Alg import Alg


class OperonAlg(Alg):
    def __init__(self, max_generation, up_op_list, down_op_list, eva_op_list, error_tolerance, population):
        super().__init__(max_generation, up_op_list, down_op_list, eva_op_list, error_tolerance, population)

    def run(self):
        pass