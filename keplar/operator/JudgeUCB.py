import math

from keplar.operator.operator import Operator


class KeplarJudgeUCB(Operator):
    def __init__(self, data_sum,subregion_num):
        super().__init__()
        self.subregion_num = subregion_num
        self.n_cluster = subregion_num
        self.lbd = None
        self.abRockNum = None
        self.abRockSum = None
        self.rockBestFit = None
        self.ucbVal = None
        self.data_sum = data_sum

    def pos_do(self):
        for i in range(self.n_cluster):
            self.ucbVal[i] = 1 / (self.rockBestFit[i] + 1) + self.lbd * math.sqrt(
                math.log(self.abRockSum) / (self.abRockNum[i]))
        self.ucbVal.sort(reversed=True)
