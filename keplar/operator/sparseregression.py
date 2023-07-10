import math
import random

import numpy as np
from sklearn.linear_model import Lasso

from sklearn.metrics import mean_squared_error

from TaylorGP.src.taylorGP.subRegionCalculator import CalFitness
from keplar.operator.operator import Operator


# class SparseRegression(Operator):
#     def __init__(self, n_cluster, tops, dataSets):
#         super().__init__()
#         self.n_cluster = n_cluster
#         self.tops = tops
#         self.dataSets = dataSets
#
#     def do(self, population=None):
#         raise NotImplementedError


class KeplarSpareseRegression(Operator):
    def __init__(self, n_cluster, ind_list, fit_list, dataSets, func_fund_num=5):
        super().__init__()
        self.func_fund_num = func_fund_num
        self.dataSets = dataSets
        self.fit_list = fit_list
        self.ind_list = ind_list
        self.n_cluster = n_cluster
        self.BestClusters = None
        self.globalBestLassoCoef = None
        self.curLassoCoef = None
        self.bestLassoFitness = None

    def do(self, population=None):
        func_fund_list = []
        for i in range(self.func_fund_num):
            func_fund = []
            for ind_arr in self.ind_list:
                index = math.floor(random.random() * len(ind_arr))
                func_fund.append(ind_arr[index])
            func_fund_list.append(func_fund)
        # print(func_fund_list)
