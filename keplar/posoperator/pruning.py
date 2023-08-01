from keplar.posoperator.posoperator import PosOperator


class MOperonPruning(PosOperator):
    def __init__(self, fit_list, n_cluster, now_repeat, repeat_limit, programs, op_list, abRockSum, abRockNum):
        super().__init__()
        self.abRockNum = abRockNum
        self.abRockSum = abRockSum
        self.op_list = op_list
        self.programs = programs
        self.n_cluster = n_cluster
        self.repeat_limit = repeat_limit
        self.now_repeat = now_repeat
        self.fit_list = fit_list

    def pos_do(self):
        if self.n_cluster > 3 and self.now_repeat > int(self.repeat_limit / 2):
            print("进行剪枝操作")
            bad_index = 0
            for i in range(len(self.fit_list)):
                if self.fit_list[bad_index][0] < self.fit_list[i][0]:
                    bad_index = i
                    self.n_cluster -= 1
                    self.fit_list.__delitem__(bad_index)
                    self.op_list.__delitem__(bad_index)
                    self.fit_list.__delitem__(bad_index)
                    self.abRockSum -= self.abRockNum[bad_index]
                    self.abRockNum.__delitem__(bad_index)
        print("不进行剪枝操作")
