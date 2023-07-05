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
from keplar.translator.translator import trans_taylor_program,taylor_trans_population



class TayloGPAlg(Alg):
    def __init__(self, generation,taylorGP_pre1,taylorGP_pre2, selector,creator,crossover,mutation,method_probs,evaluator):
        self.generation=generation
        self.taylorGP_pre1=taylorGP_pre1
        self.taylorGP_pre2=taylorGP_pre2
        self.selector=selector
        self.creator=creator
        self.crossover=crossover
        self.mutation=mutation
        self.method_probs=method_probs
        self.evaluator=evaluator

     

    def run(self):
        X, Y, qualified_list = self.taylorGP_pre1.do()
        self.taylorGP_pre2.get_value(X, Y, qualified_list)
        X,y,params,population_size,seeds,qualified_list,function_set,n_features= self.taylorGP_pre2.do()

        for i in range(self.generation):
            programs = []
            if i==0:
                population = self.creator.do()
            else:
                print(population.pop_size)
                for j in range(population.pop_size):
                    random_state = check_random_state(j)
                    method = random_state.uniform()
                    # program  = trans_taylor_program(population.target_pop_list[j])
                    self.selector.get_value(random_state,tournament_size=50)
                    pop_parent,pop_best_index = self.selector.do(population)

                    if method < self.method_probs[0]:
                        pop_honor,honor_best_index = self.selector.do(population)
                        self.crossover.get_value(random_state,qualified_list,function_set,n_features,pop_parent,pop_honor,j)
                        population= self.crossover.do(population)

                    elif method < self.method_probs[1]:
                        self.mutation.get_value(1, random_state, qualified_list, function_set, n_features, pragram_useless, pop_parent, j)
                        self.mutation.do(population)

                    elif method < self.method_probs[2]:
                        self.mutation.get_value(2, random_state, qualified_list, function_set, n_features, pragram_useless, pop_parent, j)
                        self.mutation.do(population)
                    
                    elif method < self.method_probs[3]:
                        self.mutation.get_value(3, random_state, qualified_list, function_set, n_features, pragram_useless, pop_parent, j)
                        self.mutation.do(population)

                    else:
                        self.mutation.get_value(4, random_state, qualified_list, function_set, n_features, pragram_useless, pop_parent, j)
                        self.mutation.do(population)

                    self.evaluator.do(population)
                
                
                r_best_index = population.get_tar_best()
                r_best = population.target_pop_list[r_best_index]
                r_best_fintness = population.target_fit_list[r_best_index]
                # print("r_best_index: %d"%(r_best_index))
                # print("r_best: %s"%(r_best.program.__str__))
                # print("r_best_fintness: %f"%(r_best_fintness))

        for j in range(population.pop_size):
            print(str(population.target_pop_list[j].__str__()))
            print(population.target_pop_list[j].fitness_)
        print("finished!")
                    



                    # programs.append(program)

            # self.creator.get_value(X,y,params,i,population_size,program)
            # population,pragram_useless = self.creator.do()
            # random_state = check_random_state(1)
            # pop_honor,honor_best_index = self.selector.do(population)

            
                







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
