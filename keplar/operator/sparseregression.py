import math
import random

import numpy as np
from sklearn.linear_model import Lasso

from sklearn.metrics import mean_squared_error

from TaylorGP.src.taylorGP.subRegionCalculator import CalFitness
from keplar.operator.evaluator import MetricsBingoEvaluator
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
    def __init__(self, n_cluster, ind_list, fit_list, dataSets, func_fund_num=488):
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
        self.bestLassoFitness = self.fit_list[0][0]
        for i in self.fit_list:
            if i[0] < self.bestLassoFitness:
                self.bestLassoFitness = i[0]
        Y = self.dataSets.get_np_y()
        func_fund_list = []
        for i in range(self.func_fund_num):
            func_fund = []
            for ind_arr in self.ind_list:
                index = math.floor(random.random() * len(ind_arr))
                func_fund.append(ind_arr[index])
            func_fund_list.append(func_fund)
        eval = MetricsBingoEvaluator(self.dataSets, func_fund_list=func_fund_list)
        xtrain = eval.do()
        # print(xtrain)
        # print(func_fund_list)
        if self.n_cluster <= 5:
            alphas = [0, 0.2, 0.3, 0.6, 0.8, 1.0]
        else:
            alphas = [0.2, 0.3, 0.6, 0.8, 1.0]
        for alpha in alphas:
            print(xtrain)
            print(Y)
            print(xtrain.shape)
            print(Y.shape)
            lasso_ = Lasso(alpha=alpha).fit(xtrain, Y)
            Y_pred = lasso_.predict(xtrain)
            rmseFitness = mean_squared_error(Y_pred, Y)
            self.curLassoCoef = lasso_.coef_
            if rmseFitness < self.bestLassoFitness:
                self.bestLassoFitness = rmseFitness
                self.globalBestLassoCoef = self.curLassoCoef
                if self.bestLassoFitness != float("inf"):
                    print("结果提升为: ", self.bestLassoFitness)
        print("最终适应度为"+str(self.bestLassoFitness))

