from keplar.Algorithm.Alg import Alg
from keplar.operator.creator import OperonCreator


class OperonAlg(Alg):
    def __init__(self, max_generation, up_op_list, down_op_list, eva_op_list,
                 error_tolerance, population, np_x, np_y, minL=1, maxL=50, maxD=10, decimalPrecision=5):
        super().__init__(max_generation, up_op_list, down_op_list, eva_op_list, error_tolerance, population)
        self.minL = minL
        self.maxL = maxL
        self.maxD = maxD
        self.decimalPrecision = decimalPrecision
        self.np_x = np_x
        self.np_y = np_y

    def run(self):
        self.population = OperonCreator("balanced", self.np_x, self.np_y, self.population.pop_size
                                        , "Operon", self.minL, self.maxL, self.maxD, self.decimalPrecision)
