import random
import sys
import time
import argparse

import numpy as np
from numpy import shape
from sympy import symbols

from TaylorGP.src.taylorGP._global import set_value, _init
from TaylorGP.src.taylorGP.subRegionCalculator import subRegionCalculator
from keplar.Algorithm.Alg import Alg
from TaylorGP.src.taylorGP.utils import check_random_state



# from keplar.operator.creator import OperonCreator
# from 

class TayloGPAlg(Alg):
    def __init__(self, generation, selector,creator,crossover,mutation,method_probs):
        self.generation=generation
        self.selector=selector
        self.creator=creator
        self.crossover=crossover
        self.mutation=mutation
        self.method_probs=method_probs

     

    def run(self):
        for i in range(self.generation):
            population,pragram_useless = self.creator.do()
            random_state = check_random_state(1)
            pop_parent,pop_best_index = self.selector.do(population)
            pop_honor,honor_best_index = self.selector.do(population)
            method = random_state.uniform()

            if method < self.method_probs[0]:
                population= self.crossover.do(population)
            elif method < self.method_probs[1]:
                







class MTaylorGPAlg(Alg):
    def __init__(self, max_generation, ds, up_op_list=None, down_op_list=None, eval_op_list=None, error_tolerance=None,
                 population=None,
                 recursion_limit=300, repeat=1, originalTaylorGPGeneration=20, SR_method="gplearn"):
        super().__init__(max_generation, up_op_list, down_op_list, eval_op_list, error_tolerance, population)
        self.SR_method = SR_method
        self.ds = ds
        self.originalTaylorGPGeneration = originalTaylorGPGeneration
        self.repeat = repeat
        self.recursion_limit = recursion_limit
        x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24, x25, x26, x27, x28, x29 = symbols(
            "x0,x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16,x17,x18,x19,x20,x21,x22,x23,x24,x25,x26,x27,"
            "x28,x29 ")

        set_value('_x',
                  [x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21,
                   x22, x23,
                   x24, x25, x26, x27, x28, x29])

    def run(self):
        np_x = self.ds.get_np_x()
        np_y = self.ds.get_np_y()
        np_y = np_y.reshape([-1, 1])
        self.ds = np.hstack([np_x, np_y])
        sys.setrecursionlimit(self.recursion_limit)
        argparser = argparse.ArgumentParser()
        argparser.add_argument('--fileNum', default=1, type=int)
        args = argparser.parse_args()
        dataSets = self.ds
        print("维度： ", dataSets.shape[1] - 1)
        repeat = self.repeat
        totalGeneration = self.max_generation
        originalTaylorGPGeneration = self.originalTaylorGPGeneration
        Pop = self.population.pop_size  # 种群规模
        epsilons = [1e-5, 0.2, 1, 4, 10, 100]
        # time_start1 = time.time()
        for repeatNum in range(repeat):
            # time_start2 = time.time()
            SRC = subRegionCalculator(dataSets, originalTaylorGPGeneration, mabPolicy="Greedy")
            countAvailableParameters = SRC.CalCountofAvailableParameters(epsilons=epsilons)
            mabLoopNum = max(totalGeneration // originalTaylorGPGeneration // countAvailableParameters, 1)
            for epsilon in epsilons:
                if epsilon == 1e-5:
                    SRC.PreDbscan(epsilon, noClusterFlag=True, clusterMethod="NOCLUSTER")  # 执行 OriginalTaylorGP
                elif not SRC.PreDbscan(epsilon, clusterMethod="DBSCAN"):
                    print("聚类有问题")
                    continue
                SRC.firstMabFlag = True
                _init()
                set_value('FIRST_EVOLUTION_FLAG', True)  # 进行每轮数据集演化前执行
                for tryNum in range(mabLoopNum):
                    SRC.CalTops(repeatNum, Pop, SR_method=self.SR_method)
                    SRC.SubRegionPruning()
                    SRC.SparseRegression()
                    if SRC.bestLassoFitness < 1e-5:
                        print("Final Fitness", SRC.bestLassoFitness, " Selected SubRegon Index: ",
                              SRC.globalBestLassoCoef)
                        exit()
                print("Temp Final Fitness", SRC.bestLassoFitness, " Selected SubRegon Index: ", SRC.globalBestLassoCoef)
            print("Final Fitness", SRC.bestLassoFitness, " Selected SubRegon Index: ", SRC.globalBestLassoCoef)

    """
        for fileNum in range(19,20):
            fileName = "D:\PYcharm_program\Test_everything\Bench_0.15\BenchMark_" + str(fileNum) + ".tsv"
            dataSets = np.loadtxt(fileName,dtype=np.float,skiprows=1)
            TaylorGP2Master(dataSets)    
        """
