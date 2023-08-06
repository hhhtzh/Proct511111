import math
import re

import numpy as np
import random
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import mean_squared_error

from keplar.operator.statistic import TaylorStatistic
from keplar.translator.translator import is_float
from ._global import get_value, set_value
from .originalTaylorGP import OriginalTaylorGP
from sympy import sympify
from sklearn.linear_model import Lasso


def CalFitness(eq, dataSet, IsPred=False):
    X, Y = np.split(dataSet, (-1,), axis=1)
    sympyEq = sympify(eq)
    _X = []
    _x = get_value('_x')
    len = X.shape[1]
    for i in range(len):
        X, temp = np.split(X, (-1,), axis=1)
        temp = temp.reshape(-1)
        _X.extend([temp])
    _X.reverse()
    Prediction = _calY(_x, sympyEq, X=_X)
    if IsPred: return Prediction
    try:
        rmseFitness = mean_squared_error(Prediction, Y, squared=False)
    except BaseException:
        rmseFitness = float("inf")
    return rmseFitness


def _calY(_x, f, X=None):
    y_pred = []
    len2 = len(X)
    len1 = X[0].shape[0]
    for i in range(len1):
        _sub = {}
        for j in range(len2):
            _sub.update({_x[j]: X[j][i]})
        y_pred.append(f.evalf(subs=_sub))
    return y_pred
    # fitness = Cal(self.tops[index][1])


class subRegionCalculator:
    name = 'Calculate Subregions things and MAB parameters'

    def __init__(self, dataSets, originalTaylorGPGeneration, mabPolicy="Greedy", lbd=1, NewSparseRegressionFlag=False):
        self.isolate_symbol_list = None
        self.NewSparseRegressionFlag = NewSparseRegressionFlag
        self.final_dict = None
        self.dict_arr = None
        self.dataSets = dataSets
        self.subRegions = None
        self.tops = None  # self.tops = [[subtops]] top = [[fits],[eqs]]
        self.wait2Merge = []
        self.wait2Delete = []
        self.X, self.Y = np.split(self.dataSets, (-1,), axis=1)
        self.originalTaylorGPGen = originalTaylorGPGeneration
        self.bestLassoFitness = float("inf")  # 正无穷
        self.globalBestLassoCoef = 0
        self.curBestLassoCoef = 0
        self.globalBest_X_Y_pred = []
        self.mabPolicy = mabPolicy
        self.oneSubRegionFlag = False

        # 下面是输入赌臂机相关参数，在完成第一轮的种群演化后进行MAB更新
        self.firstMabFlag = True  # 第一次执行MAB还没有最优个体反馈，选中所有臂执行TaylorGP
        self.mabArmCount = 1  # 摇臂数量就是子块的数量
        self.mabRewardFunc = None  # 奖励函数
        self.mabEpsilonProb = 0.5  # 初始Epsilon
        # 下面是自带赌臂机相关参数:mab表示整体上的，ab表示个人的
        self.mabTotalRewards = 0
        self.abAvgRewards = [0] * self.mabArmCount
        self.abRockNum = [0] * self.mabArmCount
        self.abSelectedArm = []
        self.abRockSum = 0
        self.lbd = lbd  # UCB参数
        self.ucbVal = []
        self.rockBestFit = []

    def PreDbscan(self, epsilon, data_x, n_clusters_=-1, noClusterFlag=False, clusterMethod="DBScan"):
        """
        使用DBScan密度聚类做数据集分割
        Args:
            X_Y: 原始数据集
        Returns:subRegions 分割后的numpy结果:list(numpy,numpy)

        """
        # one_flag = False #聚类结果为1的只统计一次
        # for epsilon in [0.2, 0.5, 0.8, 1, 1.5, 2, 2.5, 3, 4, 5, 10, 100]:
        # epsilon = 4  # 先固定调通代码，后续再回复for循环
        # mul_subRegions = []
        if clusterMethod == "NOCLUSTER":  # 相当于不进行分块=TaylorGP1
            self.subRegions = [self.dataSets]
            print("原始数据聚类1块哦")
        else:
            labels = None
            if clusterMethod == "DBSCAN":
                db = DBSCAN(eps=epsilon, min_samples=2 * data_x.shape[1]).fit(data_x)
                core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
                core_samples_mask[db.core_sample_indices_] = True
                labels = db.labels_  # 记录了每个数据点的分类结果，根据分类结果通过np.where就能直接取出对应类的所有数据索引了
                # Number of clusters in labels, ignoring noise if present.
                n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
                print("原始数据聚类", n_clusters_, "块")

                # 聚类过多或过少则舍掉,只统计一次聚类为1的结果
                if n_clusters_ == 1:
                    if not self.oneSubRegionFlag:
                        self.oneSubRegionFlag = True
                        self.subRegions = [self.dataSets]
                    else:
                        return False
                elif n_clusters_ < 1:
                    return False
                n_noise_ = list(labels).count(-1)

                if 1.0 * n_noise_ / self.dataSets.shape[0] > 0.5 and self.dataSets.shape[1] != 2 \
                        and self.subRegions is None:
                    return False
            elif clusterMethod == "KMEANS":
                k_means = KMeans(n_clusters=n_clusters_, random_state=10).fit(self.dataSets)
                labels = k_means.labels_
            for cluster in range(0, n_clusters_):
                dataIndex = np.where(labels == cluster * 1.0)[0]  # 保留所有分类是第一类（第二类是1.0）的数据集序号
                for i in range(0, dataIndex.shape[0]):  # 反推当前子块索引对应数据
                    if i == 0:
                        arr = np.array([self.dataSets[dataIndex[0]]])  # 升维
                    else:
                        arr = np.append(arr, np.array([self.dataSets[dataIndex[i]]]), axis=0)
                if cluster == 0:
                    self.subRegions = [arr]  # 合并所有子块的数据
                else:
                    self.subRegions.extend([arr])
        self.tops = [0] * len(self.subRegions)
        self.countOfNotYetEvolvedSubRegions = len(self.subRegions)
        # 更新MAB参数
        self.mabArmCount = len(self.subRegions)
        self.abAvgRewards = [0] * self.mabArmCount
        self.abRockNum = [0] * self.mabArmCount
        self.ucbVal = [0] * self.mabArmCount
        self.rockBestFit = [1e+5] * self.mabArmCount
        return True

        # data_res = [X_Y[data_res[i]] for i in range(data_res.shape[0])]
        # Data_res = np.where(labels == 1.0)  # 保留所有分类是第二类（第一类是1.0）的数据集序号

    def CalArmIndexOfBestIndividudual(self):
        res = []
        if self.mabPolicy == "Greedy":
            # 系数小于1e-5的不参与下一轮演化
            for coef in self.curLassoCoef:
                if abs(coef) > 1e-5:
                    res.append(1)
                else:
                    res.append(0)
        elif self.mabPolicy == "UCB":
            # 选择UCB值最大的子区间进行演化:ucbs[0]存储各个子区间的ucb值，[1]存储各子区间所代表结果的最优适应度，##[2]存储各子区间被选中次数，[3]表示总次数
            for i in range(len(self.ucbVal)):
                self.ucbVal[i] = 1 / (self.rockBestFit[i] + 1) + self.lbd * math.sqrt(
                    math.log(self.abRockSum) / (self.abRockNum[i]))
            maxUCB = max(self.ucbVal)
            for ucb in self.ucbVal:
                if abs(ucb - maxUCB) < 1e-5:
                    res.append(1)
                else:
                    res.append(0)
                # self.abRockNum[i] +=1 #后面统一加1了
                # self.abRockSum += 1
        return res

    def CalTaylorRmseOfSubRegions(self, repeatNum):
        # 计算各个子块对应的Taylor展开式拟合误差
        totalNmse = 0
        for selectedRegionIndex in range(len(self.subRegions)):  # 计算各子块的展开式拟合误差
            # top1 = OriginalTaylorGP(dataSets,repeatNum)#使用原始数据集测试代码流程是否正常
            # print(" selectedRegionIndex",selectedRegionIndex)
            try:
                totalNmse += OriginalTaylorGP(self.subRegions[selectedRegionIndex], None, None, repeatNum, 0, 100,
                                              rmseFlag=True)  # top是list 0是适应度，1是公式 2是上轮最后一代种群
            except BaseException:
                print("OriginalTaylorGP Error!")
                return float("inf")
        avgNmse = totalNmse / len(self.subRegions)
        print("Count of SubRegions: ", len(self.subRegions), "Avg Nmse:", "%.5g" % avgNmse)
        return avgNmse

    def CalTops(self, repeatNum, Pop, SR_method="gplearn"):
        """
        对选中的子区间分别执行OriginalTaylorGP
        Args:
            repeatNum: 作为随机数种子
            Pop: 进化量
            :param SR_method:
        """
        print(self.firstMabFlag)
        print(self.mabPolicy)
        if self.firstMabFlag:
            self.abSelectedArm = [1] * self.mabArmCount
            self.firstMabFlag = False
        elif self.mabPolicy == "Greedy":
            if random.random() < self.mabEpsilonProb:
                print("更新所有索引")
                self.abSelectedArm = [1] * self.mabArmCount
            else:
                print("更新被选中索引")
                self.abSelectedArm = self.CalArmIndexOfBestIndividudual()
                print(self.abSelectedArm)
        elif self.mabPolicy == "Greedy1":
            pass
        elif self.mabPolicy == "UCB":
            self.abSelectedArm = self.CalArmIndexOfBestIndividudual()
            pass
        elif self.mabPolicy == "NoMAB":
            self.abSelectedArm = [1] * self.mabArmCount
        # tops = []  # 存储m个聚类的前k个个体，先测试一个
        for selectedRegionIndex in [j for j, x in enumerate(self.abSelectedArm) if
                                    x == 1]:  # 目前是使用串行演化，先这样不改了，后面看情况是否改为并行
            # top1 = OriginalTaylorGP(dataSets,repeatNum)#使用原始数据集测试代码流程是否正常
            print("len(self.subRegions): ", len(self.subRegions), " selectedRegionIndex", selectedRegionIndex)
            if len(self.subRegions) <= selectedRegionIndex:
                break
            self.abRockNum[selectedRegionIndex] += 1
            self.abRockSum += 1
            parents, qualified_list, Y_pred = None, None, None

            if self.abSelectedArm.count(1) > 1:
                print("Pop,self.abSelectedArm.count(1),self.abSelectedArm", Pop, self.abSelectedArm.count(1),
                      self.abSelectedArm, sep=" ")
                Pop = max(Pop // self.abSelectedArm.count(1), 10)  # 种群大小至少为10
                # [end_fitness, programs, population, findBestFlag, qualified_list, Y_pred]
            top1 = OriginalTaylorGP(self.subRegions[selectedRegionIndex], Y_pred, parents, repeatNum,
                                    self.originalTaylorGPGen, Pop, qualified_list=qualified_list,
                                    SR_method=SR_method)  # top是list 0是适应度，1是公式 2是上轮最后一代种群
            self.tops[selectedRegionIndex] = top1  # 由于MAB，所以选择性更新tops
            if not get_value('FIRST_EVOLUTION_FLAG'):  # 除去第一次，以后演化基于之前的父代，并且若不不存在父代说明是低阶多项式不用演化直接跳过，此处也不影响MAB
                # print(self.tops)
                # print(len(self.tops))
                # print(selectedRegionIndex)
                # if len(self.tops) != 1:
                #     if isinstance(self.tops[1], int):
                #         self.tops = self.tops[0]
                # print(self.tops)
                # print(len(self.tops))
                # print(selectedRegionIndex)
                # if len(self.tops) == 1:
                subRegionFindBestFlag = self.tops[selectedRegionIndex][3]
                qualified_list = self.tops[selectedRegionIndex][4]
                Y_pred = self.tops[selectedRegionIndex][5]
                # else:
                #     print(self.tops[1])
                #     subRegionFindBestFlag = self.tops[0][3]
                #     qualified_list = self.tops[0][4]
                #     Y_pred = self.tops[0][5]

                if not subRegionFindBestFlag:
                    parents = self.tops[selectedRegionIndex][2]
                else:
                    continue

        set_value('FIRST_EVOLUTION_FLAG', False)
        return self.tops

    def SubRegionPruning(self):
        """
          1.暂时去掉此观点-->合并Taylor 特征相同的子块，同时合并赌臂机参数
          2.临块BestInd优于当前块BestInd，删掉当前块(低效)，并调整赌臂机参数
          实现：将问题转换为删除待合并子块Flag或待删除子块Flag为True的subRegions和对应tops
          3.更新子块相关参数
        Args:
            subRegions:聚类后的子块
            temp_tops:保存每个子块的topk个体数组
        Returns:
            tops:更新后的子块与对应topk个体
        """
        print("Pruning")
        # print("subregion:" + str(self.subRegions))
        for i in range(len(self.subRegions)):
            print(i)
            # eq_str = str(self.tops[i][1][0])
            # print("pruning_eq_str:" + str(eq_str))
        pruningFlag = False
        for i in range(len(self.subRegions) - 1, 0, -1):  # 从len-1到0，左闭右开
            try:
                # if self.final_dict is not None:
                if self.EvaluateNearRegionFitness(i - 1, i):  # 使用 i-1块的最优个体测试i块
                    self.DelRegionParameters(i)
                    pruningFlag = True
                elif self.EvaluateNearRegionFitness(i, i - 1):  # 使用 i块的最优个体测试i-1块
                    self.DelRegionParameters(i - 1)
                    pruningFlag = True

            except BaseException:
                print("评估临块Error")
        if pruningFlag:
            print("Having pruning")
        else:
            print("NO pruning")

    def NewPruning(self):
        pass

    def DelRegionParameters(self, index):
        """
        删除子块中包含第i块的所有信息，防止影响到后面块的现有信息
        Args:
            index: subRegion索引
        """
        del self.subRegions[index], self.tops[index]
        del self.abAvgRewards[index], self.abRockNum[index]
        if self.mabPolicy == "UCB": del self.ucbVal[index]

    def EvaluateNearRegionFitness(self, BestEqindex=-1, nearRegionIndex=-1):
        """
        评估当前块的最优个体是否比相邻块的适应度更优：先转成sympy公式再计算RMSE
        Args:
            index:当前子块的索引
        Returns:
            fitness
        """
        eq = self.tops[BestEqindex][1][0]
        nearRegionData = self.subRegions[nearRegionIndex]
        # X, Y = np.split(nearRegionData, (-1,), axis=1)
        rmseFitness = CalFitness(eq, nearRegionData)
        print(rmseFitness, self.tops[nearRegionIndex][0][0])
        return rmseFitness < self.tops[nearRegionIndex][0][0]

    def SparseRegression(self):
        """
        将各子块的top3个体按7:2:1的概率执行稀疏回归-->改为随机+最优 组合
        Returns:
            bestIndividualIndex:最优个体所在的子块索引
        """
        """
        if len(self.subRegions) == 1:
            self.mabPolicy = "None"
            print("只有一个分区也可以LR！")
            return 
        else:        
        """
        if len(self.subRegions) == 1:
            self.bestLassoFitness = self.tops[0][0][0]
            self.curLassoCoef = [1]
            self.globalBestLassoCoef = [1]
            self.BestClusters = 1
            X, Y = np.split(self.dataSets, (-1,), axis=1)
            self.globalBest_X_Y_pred = [[X, self.tops[0][5]]]
            return
        X_trains = self.CalXOfLasso()  # 将各子块的top3个体按7:2:1的概率执行稀疏回归-->改为最优+随机 组合
        if self.NewSparseRegressionFlag:
            final_dict = {}
            for i in self.dict_arr:
                for key in i:
                    if key not in final_dict:
                        final_dict.update({key: i[key]})
                    else:
                        now_num = final_dict[key]
                        now_num += i[key]
                        final_dict.update({key: now_num})
            isolate_symbol_list = []
            for key in final_dict:
                str_eq = str(final_dict[key]) + "*" + str(key)
                isolate_symbol_list.append(str_eq)
            self.isolate_symbol_list = isolate_symbol_list
            X_trains = self.NewCalLasso()
            try:
                for X_train in X_trains:
                    if not X_train: continue
                    if len(self.subRegions) <= 5:
                        alphas = [0, 0.2, 0.3, 0.6, 0.8, 1.0]
                    else:
                        alphas = [0.2, 0.3, 0.6, 0.8, 1.0]
                    for alpha in alphas:
                        lasso_ = Lasso(alpha=alpha).fit(X_train, self.Y)
                        Y_pred = lasso_.predict(X_train)
                        rmseFitness = mean_squared_error(Y_pred, self.Y)
                        self.curLassoCoef = lasso_.coef_
                        if rmseFitness < self.bestLassoFitness:
                            self.bestLassoFitness = rmseFitness
                            self.globalBestLassoCoef = self.curLassoCoef
                            if self.bestLassoFitness != float("inf"):
                                print("结果提升为: ", self.bestLassoFitness)
                            self.UpdateRockBestFit()
                            self.Cal_X_Y_pred()  # 更新self.globalBest_X_Y_pred
                # print("Final Fitness",self.bestLassoFitness, " Selected SubRegon Index: ", self.globalBestLassoCoef)
                self.UpdateAvgRewards()
            except BaseException:
                print("TypeError: can't convert complex to float")
                self.curLassoCoef = [1] * len(self.subRegions)  # 保证下轮对所有子块都进行更新.
            print("final::::" + str(final_dict))
            self.final_dict = final_dict
        else:
            try:
                for X_train in X_trains:
                    if not X_train: continue
                    if len(self.subRegions) <= 5:
                        alphas = [0, 0.2, 0.3, 0.6, 0.8, 1.0]
                    else:
                        alphas = [0.2, 0.3, 0.6, 0.8, 1.0]
                    for alpha in alphas:
                        lasso_ = Lasso(alpha=alpha).fit(X_train, self.Y)
                        Y_pred = lasso_.predict(X_train)
                        rmseFitness = mean_squared_error(Y_pred, self.Y)
                        print("uuuuuu")
                        print(lasso_.coef_)
                        self.curLassoCoef = lasso_.coef_
                        worest_index = 0
                        worest_flag = True
                        for i in range(len(self.curLassoCoef)):
                            worest_flag = True
                            for j in self.curLassoCoef:
                                if self.curLassoCoef[i] >= j / 2:
                                    worest_flag = False
                                    worest_index = i
                                    break
                            if worest_flag:
                                break
                        if worest_flag:
                            self.DelRegionParameters(worest_index)
                        if rmseFitness < self.bestLassoFitness:
                            self.bestLassoFitness = rmseFitness
                            self.globalBestLassoCoef = self.curLassoCoef
                            if self.bestLassoFitness != float("inf"):
                                print("结果提升为: ", self.bestLassoFitness)
                            self.UpdateRockBestFit()
                            self.Cal_X_Y_pred()  # 更新self.globalBest_X_Y_pred
                # print("Final Fitness",self.bestLassoFitness, " Selected SubRegon Index: ", self.globalBestLassoCoef)
                self.UpdateAvgRewards()

                # for i in range(len(self.globalBestLassoCoef)):
                #     for j in self.dict_arr:
                #         for key in j:
                #             j[key] = j[key] * float(self.globalBestLassoCoef[i])
                # final_dict = {}
                # for i in self.dict_arr:
                #     for key in i:
                #         if key not in final_dict:
                #             final_dict.update({key: i[key]})
                #         else:
                #             now_num = final_dict[key]
                #             now_num += i[key]
                #             final_dict.update({key: now_num})


            except BaseException:
                print("TypeError: can't convert complex to float")
                self.curLassoCoef = [1] * len(self.subRegions)  # 保证下轮对所有子块都进行更新.


    def Cal_X_Y_pred(self):
        """
        Returns:
            list: 当前聚类方式下各子块的X_Y_pred
        """
        self.globalBest_X_Y_pred = []
        for i, subRegion in enumerate(self.subRegions):
            X_sub, Y_sub = np.split(subRegion, (-1,), axis=1)
            self.globalBest_X_Y_pred.append([X_sub, self.tops[i][5]])
        return self.globalBest_X_Y_pred

    def UpdateRockBestFit(self):
        """
        更新各子区域所能实现的最优LR结果
        """
        for i, coef in enumerate(self.globalBestLassoCoef):
            if abs(coef) > 1e-5:
                self.rockBestFit[i] = self.bestLassoFitness

    def UpdateAvgRewards(self):
        """
        选中的臂参与贡献了最优个体则奖励值为1，否则为0
        """
        for i in range(len(self.abSelectedArm)):
            action = 0
            if self.abSelectedArm is not None and self.abSelectedArm[i] == 1 and self.globalBestLassoCoef[i] != 0:
                action = 1
            self.abAvgRewards[i] = (self.abAvgRewards[i] * (self.abRockNum[i] - 1) + action) / self.abRockNum[i]

    def CalXOfLasso(self):
        """
        #将各子块的top3个体按7:2:1的概率执行稀疏回归-->改为最优+随机 组合
        根据选中的个体计算其对应的训练集特征X，需要重新计算，之前只是在子块进行评估，这里需要在整个个数据集上评估
        """

        global arr
        bestFlag = True
        numberOfCombinations = 5
        X_Trains = []
        num = 100000
        dict_arr = []
        for num in range(numberOfCombinations):
            index = 0
            for i, top in enumerate(self.tops):  # self.tops = [[subtops]] top = [[fits],[eqs],[population]]
                str_eq = str(top[1][index])
                print(str_eq)
                strx_ = re.sub(r'x(\d{1})', r'x_\1', str_eq)
                print(strx_)
                sta = TaylorStatistic(strx_)
                sta.pos_do()
                dict1 = sta.final_statis
                print(dict1)
                dict_arr.append(dict1)
                if not bestFlag:
                    index = math.floor(random.random() * len(top[1]))
                try:
                    tempArr = np.array(CalFitness(eq=top[1][index], dataSet=self.dataSets, IsPred=True)).reshape(-1, 1)



                except BaseException:  # 此公式不适用于完整数据集，需要跳过
                    num += 1
                    continue
                if i == 0:
                    arr = tempArr
                else:
                    arr = np.append(arr, tempArr, axis=1)
            bestFlag = False  # 第一次按index = 0 最优个体处理，以后按随机选取
            try:
                X_Trains.extend([arr])
            except BaseException:
                print("没有找到适用于完整数据集的公式")

            self.dict_arr = dict_arr
            print("dict_arr:" + str(self.dict_arr))
            return X_Trains

    def NewCalLasso(self):
        global arr
        bestFlag = True
        numberOfCombinations = 5
        X_Trains = []
        num = 100000
        dict_arr = []
        for i, top in enumerate(self.isolate_symbol_list):
            str_eq = str(top)
            try:
                tempArr = np.array(CalFitness(eq=str_eq, dataSet=self.dataSets, IsPred=True)).reshape(-1, 1)
            except BaseException:  # 此公式不适用于完整数据集，需要跳过
                num += 1
                continue
            if i == 0:
                arr = tempArr
            else:
                arr = np.append(arr, tempArr, axis=1)
            bestFlag = False  # 第一次按index = 0 最优个体处理，以后按随机选取
            try:
                X_Trains.extend([arr])
            except BaseException:
                print("没有找到适用于完整数据集的公式")
            return X_Trains

    def MAB(self, bestIndividualIndex, policy="epsilonGreedy"):
        """
        根据多赌臂机策略进行子块选择性演化
        Args:
            bestIndividualIndex:最优个体所在的子块索引
            policy:Default=“epsilonGreedy”，可选UCB，AB-MAB
        Returns:
            updatingRegion:最终决定更新的子块序列

        """

    def CalCountofAvailableParameters(self, np_x, epsilons=None, clusters=None):
        count = 0
        if epsilons is not None:
            for epsilon in epsilons:
                if epsilon == 1e-5: continue
                if self.PreDbscan(epsilon, clusterMethod="DBSCAN", data_x=np_x):
                    count += 1
            return count + 1
        elif clusters is not None:
            return len(clusters)
