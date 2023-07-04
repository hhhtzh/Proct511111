import random
import sys
import time
import argparse

import numpy as np
from numpy import shape

from TaylorGP.src.taylorGP._global import set_value
from TaylorGP.src.taylorGP.subRegionCalculator import subRegionCalculator
from keplar.Algorithm.Alg import Alg


# from keplar.operator.creator import OperonCreator
# from 

class TayloGPAlg(Alg):
    # def __init__(self, max_generation, up_op_list, down_op_list, eva_op_list, error_tolerance, population):
    #     super().__init__(max_generation, up_op_list, down_op_list, eva_op_list, error_tolerance, population)
    def __init__(self, max_generation, up_op_list, down_op_list, eval_op_list, error_tolerance, population):
        # super().__init__()
        super().__init__(max_generation, up_op_list, down_op_list, eval_op_list, error_tolerance, population)

    def run(self):
        # return super().run()
        # return super().run()
        print("done!")


class MTaylorGPAlg(Alg):
    def __init__(self, max_generation, ds,up_op_list=None, down_op_list=None, eval_op_list=None, error_tolerance=None, population=None,
                 recursion_limit=300, repeat=3, originalTaylorGPGeneration=20,SR_method="gplearn"):
        super().__init__(max_generation, up_op_list, down_op_list, eval_op_list, error_tolerance, population)
        self.SR_method = SR_method
        self.ds = ds
        self.originalTaylorGPGeneration = originalTaylorGPGeneration
        self.repeat = repeat
        self.recursion_limit = recursion_limit

    def run(self):
        np_x=self.ds.get_np_x()
        np_y=self.ds.get_np_y()
        np_y = np_y.reshape([-1, 1])
        self.ds=np.hstack([np_x, np_y])
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
                set_value('FIRST_EVOLUTION_FLAG', True)  # 进行每轮数据集演化前执行
                for tryNum in range(mabLoopNum):
                    SRC.CalTops(repeatNum, Pop,SR_method=self.SR_method)
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
