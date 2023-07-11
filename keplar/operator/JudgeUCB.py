import math

from keplar.operator.operator import Operator


class KeplarJudgeUCB(Operator):
    def __init__(self, subregion_num, abRockSum, abRockNum):
        super().__init__()
        self.max_ucb_index = None
        self.max_ucb = None
        self.abRockNum = abRockNum
        self.abRockSum = abRockSum
        self.n_cluster = subregion_num
        self.lbd = None
        self.rockBestFit = None
        self.ucbVal = None

    def pos_do(self):
        for i in range(self.n_cluster):
            # self.ucbVal[i] = 1 / (self.rockBestFit[i] + 1) + self.lbd * math.sqrt(
            self.ucbVal[i] = 1 / (self.rockBestFit[i] + 1) + math.sqrt(
                math.log(self.abRockSum) / (self.abRockNum[i]))
        self.max_ucb = self.ucbVal[0]
        self.max_ucb_index = 0
        for i in range(self.n_cluster):
            if self.ucbVal[i] > self.max_ucb:
                self.max_ucb = self.ucbVal[i]
                self.max_ucb_index = i
        print("最大UBC:" + str(self.ucbVal[0]) + "最大UCB子空间index" + str(self.max_ucb_index))
