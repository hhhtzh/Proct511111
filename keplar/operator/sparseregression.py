import math
import random

import numpy as np
from sklearn.linear_model import Lasso, LassoCV

from sklearn.metrics import mean_squared_error
from sklearn.svm._libsvm import predict

from keplar.operator.evaluator import MetricsBingoEvaluator
from keplar.operator.operator import Operator
from keplar.operator.predictor import MetricsBingoPredictor
from keplar.operator.statistic import BingoStatistic


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
        self.final_str_ind = None
        self.lasso_list = None
        self.rockBestFit = [1e+5] * n_cluster
        self.func_fund_num = func_fund_num
        self.dataSets = dataSets
        self.fit_list = fit_list
        self.ind_list = ind_list
        self.n_cluster = n_cluster
        self.BestClusters = None
        self.globalBestLassoCoef = None
        self.curLassoCoef = []
        self.bestLassoFitness = None

    def do(self, population=None):
        self.lasso_list = []
        self.bestLassoFitness = self.fit_list[0][0]
        for i in self.fit_list:
            if i[0] < self.bestLassoFitness:
                self.bestLassoFitness = i[0]
        Y = self.dataSets.get_np_y()
        func_fund_list = []
        for i in range(self.n_cluster):
            func_fund = []
            for ind_arr in self.ind_list:
                index = math.floor(random.random() * len(ind_arr))
                func_fund.append(ind_arr[index])
            func_fund_list.append(func_fund)
        pred = MetricsBingoPredictor(self.dataSets, func_fund_list=func_fund_list)
        xtrain = pred.do()
        # print(xtrain)
        np_xtrain = np.array(xtrain)
        print(np.shape(np_xtrain))
        # print(func_fund_list)
        if self.n_cluster <= 5:
            alphas = [0, 0.2, 0.3, 0.6, 0.8, 1.0]
        else:
            alphas = [0.2, 0.3, 0.6, 0.8, 1.0]
        for alpha in alphas:
            lasso_ = Lasso(alpha=alpha).fit(xtrain, Y)
            Y_pred = lasso_.predict(xtrain)
            rmseFitness = mean_squared_error(Y_pred, Y)
            self.curLassoCoef = lasso_.coef_
            self.lasso_list.append(rmseFitness)
            if rmseFitness < self.bestLassoFitness:
                self.bestLassoFitness = rmseFitness
                self.globalBestLassoCoef = self.curLassoCoef
                if self.bestLassoFitness != float("inf"):
                    print("结果提升为: ", self.bestLassoFitness)
                for i, coef in enumerate(self.globalBestLassoCoef):
                    if abs(coef) > 1e-5:
                        self.rockBestFit[i] = self.bestLassoFitness
        # print(self.globalBestLassoCoef)
        # for i, coef in enumerate(self.globalBestLassoCoef):
        print(self.globalBestLassoCoef)
        print("最好子空间适应度为" + str(self.bestLassoFitness))
        print("coef:" + str(self.curLassoCoef))
        final_str_ind = ""
        print("ind_list" + str(self.ind_list))
        dict_arr = []
        for i in range(len(self.globalBestLassoCoef)):
            str_equ = str(self.ind_list[i][0])
            sta = BingoStatistic(str_equ)
            sta.pos_do()
            dict1 = sta.final_statis
            for key in dict1:
                dict1[key] = dict1[key] * float(self.globalBestLassoCoef[i])
            dict_arr.append(dict1)
            temp_str_ind = "(" + str(self.globalBestLassoCoef[i]) + "*(" + str(self.ind_list[i][0]) + ")" + ")"
            if final_str_ind != "":
                if self.globalBestLassoCoef[i] != 0:
                    final_str_ind = final_str_ind + "+" + temp_str_ind
            else:
                final_str_ind = temp_str_ind
        final_dict = {}
        for i in dict_arr:
            for key in i:
                if key not in final_dict:
                    final_dict.update({key: i[key]})
                else:
                    now_num = final_dict[key]
                    now_num += i[key]
                    final_dict.update({key: now_num})

        self.final_str_ind = final_str_ind
        # print(final_str_ind)
        print("final::::" + str(final_dict))

# class KeplarReinForceSpareRegression(Operator):
#     def __init__(self, n_cluster, ind_list, fit_list, dataSets, func_fund_num=488):
#         super().__init__()
#         self.final_str_ind = None
#         self.lasso_list = None
#         self.rockBestFit = [1e+5] * n_cluster
#         self.func_fund_num = func_fund_num
#         self.dataSets = dataSets
#         self.fit_list = fit_list
#         self.ind_list = ind_list
#         self.n_cluster = n_cluster
#         self.BestClusters = None
#         self.globalBestLassoCoef = None
#         self.curLassoCoef = []
#         self.bestLassoFitness = None

