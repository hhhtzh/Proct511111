import math

from keplar.operator.operator import Operator


class JudgeUCB(Operator):
    def __init__(self, data_sum):
        super().__init__()
        self.lbd = None
        self.abRockNum = None
        self.abRockSum = None
        self.rockBestFit = None
        self.ucbVal = None
        self.data_sum = data_sum

    def pos_do(self):
        for i in range(len(self.data_sum)):
            self.ucbVal[i] = 1 / (self.rockBestFit[i] + 1) + self.lbd * math.sqrt(
                math.log(self.abRockSum) / (self.abRockNum[i]))

        self.ucbVal.sort(reversed=True)
