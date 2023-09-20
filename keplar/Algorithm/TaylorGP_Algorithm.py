import random
import re
import sys
import time
import argparse
import copy

import numpy as np
from numpy import shape
from sympy import symbols

from TaylorGP.TaylorGP2_KMEANS import Cal_fitness_Coef
from TaylorGP.src.taylorGP._global import set_value, _init
from TaylorGP.src.taylorGP.subRegionCalculator import subRegionCalculator

from keplar.Algorithm.Alg import Alg
from TaylorGP.src.taylorGP.utils import check_random_state
from keplar.operator.evaluator import SingleBingoEvaluator
from keplar.translator.translator import trans_taylor_program, taylor_trans_population, trans_op1, trans_op2
# from TaylorGP.src.taylorGP.genetic import BaseSymbolic
from TaylorGP.src.taylorGP.fitness import _mean_square_error, _weighted_spearman, _log_loss, _mean_absolute_error, \
    _Fitness
import math


class TayloGPAlg(Alg):
    def __init__(self, generation, taylorGP_pre1, taylorGP_pre2, selector, creator, crossover, mutation, method_probs,
                 taylorsort, evaluator):
        self.generation = generation
        self.taylorGP_pre1 = taylorGP_pre1
        self.taylorGP_pre2 = taylorGP_pre2
        self.selector = selector
        self.creator = creator
        self.crossover = crossover
        self.mutation = mutation
        self.method_probs = method_probs
        self.evaluator = evaluator
        self.taylorsort = taylorsort
        self.parsimony_coefficient = 0.001
        self.population_size = 1000
        self.run_details_ = {'generation': [],
                             'average_length': [],
                             'average_fitness': [],
                             'best_length': [],
                             'best_fitness': [],
                             'best_oob_fitness': [],
                             'generation_time': []}
        self.max_samples = 1.0
        self.verbose = 1
        self.stopping_criteria = 0.0
        self.sample_weight = None
        # self.params= params
        # self.X =X
        # self.y =y 

    def print_details(self, run_details=None, i=None):
        """A report of the progress of the evolution process.

        Parameters
        ----------
        run_details : dict
            Information about the evolution.

        """
        if run_details is None:
            print('    |{:^25}|{:^42}|'.format('Population Average',
                                               'Best Individual'))
            print('-' * 4 + ' ' + '-' * 25 + ' ' + '-' * 42 + ' ' + '-' * 10)
            line_format = '{:>4} {:>8} {:>16} {:>8} {:>16} {:>16} {:>10}'
            print(line_format.format('Gen', 'Length', 'Fitness', 'Length',
                                     'Fitness', 'OOB Fitness', 'Time Left'))

        else:
            # Estimate remaining time for run
            gen = run_details['generation'][i]
            # generation_time = run_details['generation_time'][i]
            # remaining_time = (self.generations - gen - 1) * generation_time
            # if remaining_time > 60:
            #     remaining_time = '{0:.2f}m'.format(remaining_time / 60.0)
            # else:
            #     remaining_time = '{0:.2f}s'.format(remaining_time)
            remaining_time = '{0:.2f}s'.format(0.0)

            oob_fitness = 'N/A'
            # line_format = '{:4d} {:8.2f} {:16g} {:8d} {:16g} {:>16} {:>10}'
            # if self.max_samples < 1.0:
            oob_fitness = run_details['best_oob_fitness'][i]
            line_format = '{:4d} {:8.2f} {:16g} {:8d} {:16g} {:16g} {:>10}'

            print(line_format.format(run_details['generation'][i],
                                     run_details['average_length'][i],
                                     run_details['average_fitness'][i],
                                     run_details['best_length'][i],
                                     run_details['best_fitness'][i],
                                     oob_fitness,
                                     remaining_time
                                     ))

    def select_by_crowding_distance(self, population, front, reminder):
        cur_population = copy.deepcopy([population[i] for i in front])
        cur_population.sort(key=lambda x: x.raw_fitness_)
        sorted1 = copy.deepcopy(cur_population)
        cur_population.sort(key=lambda x: x.length_)
        sorted2 = cur_population
        distance = [0 for i in range(0, len(front))]
        distance[0] = 4444444444444444
        distance[len(front) - 1] = 4444444444444444
        fitness_ = [sorted1[i].raw_fitness_ for i in range(len(cur_population))]
        length_ = [sorted2[i].length_ for i in range(len(cur_population))]
        maxFit, minFit, maxLen, minLen = max(fitness_), min(fitness_), max(length_), min(length_)
        # 第k个个体的距离就是front[k]的距离----dis[k]==front[k]
        for k in range(1, len(front) - 1):
            distance[k] = distance[k] + (fitness_[k + 1] - fitness_[k - 1]) / (
                    maxFit - minFit + 0.01)
        for k in range(1, len(front) - 1):
            distance[k] = distance[k] + (length_[k + 1] - length_[k - 1]) / (
                    maxLen - minLen + 0.01)
        index_ = sorted(range(len(distance)), key=lambda k: distance[k])
        index_.reverse()
        reminderPop = [cur_population[i] for i in index_][:reminder]
        return reminderPop

    def run(self):
        X, Y, qualified_list = self.taylorGP_pre1.do()
        self.taylorGP_pre2.get_value(X, Y, qualified_list)
        X, y, params, population_size, seeds, qualified_list, function_set, n_features = self.taylorGP_pre2.do()
        parents = None

        n_samples, n_features = X.shape
        max_samples = params['max_samples']
        max_samples = int(max_samples * n_samples)

        # seeds = random_state.randint(MAX_INT, size=self.population_size)

        for gen in range(self.generation):
            programs = []
            if gen == 0:
                population, sample_weight = self.creator.do()
                self.print_details()
                # self.print_details(self.run_details_,gen-1)

            else:
                # print(population.pop_size)
                # for j in range(population.pop_size):
                j = 0
                print(population.pop_size)
                while j != population.pop_size:
                    ran = np.random.randint(0, 10000)
                    random_state = check_random_state(ran)
                    method = random_state.uniform()
                    # program  = trans_taylor_program(population.target_pop_list[j])
                    self.selector.get_value(random_state, tournament_size=50)
                    pop_parent, pop_best_index = self.selector.do(parents)

                    if method < self.method_probs[0]:
                        print(0)
                        # print("ttt")
                        # print(population.target_pop_list[j].get_expression())

                        self.selector.get_value(random_state, tournament_size=50)
                        pop_honor, honor_best_index = self.selector.do(parents)
                        # print("how")
                        self.crossover.get_value(random_state, pop_parent, pop_honor.program, j)
                        population = self.crossover.do(population)
                        # print("how2")

                        # population.target_pop_list[]

                    elif method < self.method_probs[1]:
                        # print("rrrr")
                        print(1)

                        self.mutation.get_value(1, random_state, pop_parent, j)
                        population = self.mutation.do(population)
                        # print("how2")


                    elif method < self.method_probs[2]:
                        print(2)

                        self.mutation.get_value(2, random_state, pop_parent, j)
                        population = self.mutation.do(population)

                    elif method < self.method_probs[3]:
                        print(3)

                        self.mutation.get_value(3, random_state, pop_parent, j)
                        population = self.mutation.do(population)

                    else:
                        print(4)

                        self.mutation.get_value(4, random_state, pop_parent, j)
                        population = self.mutation.do(population)

                    # curr_sample_weight = np.ones((n_samples,))
                    # gp_fit = _Fitness(_weighted_spearman, False)

                    # pred_y = program.execute(self.eval_x)
                    # population.target_pop_list[j].raw_fitness_ = gp_fit(self.eval_y, pred_y, self.feature_weight)
                    # # population.target_pop_list[j].raw_fitness_ = population.target_pop_list[j].raw_fitness(X, y, curr_sample_weight)
                    # if math.isnan(population.target_pop_list[j].raw_fitness_) or math.isinf(population.target_pop_list[j].raw_fitness_) or population.target_pop_list[j].length_ >500:
                    #     j -= 1
                    #     continue

                    if sample_weight is None:
                        curr_sample_weight = np.ones((n_samples,))
                    else:
                        curr_sample_weight = sample_weight.copy()

                    oob_sample_weight = curr_sample_weight.copy()
                    program = population.target_pop_list[j]

                    indices, not_indices = program.get_all_indices(n_samples,
                                                                   max_samples,
                                                                   random_state)

                    curr_sample_weight[not_indices] = 0
                    oob_sample_weight[indices] = 0

                    program.raw_fitness_ = program.raw_fitness(X, y, curr_sample_weight)
                    print(program.raw_fitness_)

                    if math.isnan(program.raw_fitness_) or math.isinf(program.raw_fitness_) or program.length_ > 500:
                        # i =i- 1
                        # i -= 1
                        # idx = i
                        # print(i)
                        # n_pop += 1
                        # j-=1
                        rand = random.randint(0, 1000)
                        random_state = check_random_state(rand)

                        print("math.isnan")
                        print(program.length_)
                        continue
                        # pass
                    program.fitness_ = program.fitness(self.parsimony_coefficient)

                    population.target_fit_list[j] = program.fitness_

                    # print("test111")
                    # print(program.fitness_)

                    # if max_samples < n_samples:
                    #     # Calculate OOB fitness
                    #     program.oob_fitness_ = program.raw_fitness(self.X, self.y, oob_sample_weight)

                    # if idx == i:
                    #     print("????")
                    # population.target_append(program)

                    # print(j)
                    j += 1

                fitness = [program.raw_fitness_ for program in population.target_pop_list]
                length = [program.length_ for program in population.target_pop_list]

                fitness_ = [program.fitness_ for program in population.target_pop_list]

                if self.selector.greater_is_better:
                    best_program = population.target_pop_list[np.argmax(fitness)]
                    best_program_fitness_ = population.target_pop_list[np.argmax(fitness_)]
                else:
                    best_program = population.target_pop_list[np.argmin(fitness)]
                    best_program_fitness_ = population.target_pop_list[np.argmin(fitness_)]

                self.run_details_['generation'].append(gen)
                self.run_details_['average_length'].append(np.mean(length))
                self.run_details_['average_fitness'].append(np.mean(fitness_))
                self.run_details_['best_length'].append(best_program.length_)
                self.run_details_['best_fitness'].append(best_program.fitness_)
                # oob_fitness = np.nan

            parents = population
            print(population.pop_size)

            # r_best_index = population.get_tar_best()
            # r_best = population.target_pop_list[r_best_index]
            # r_best_fintness = population.target_fit_list[r_best_index]
            # print("r_best_index: %d"%(r_best_index))
            # print("r_best: %s"%(r_best.__str__()))
            # print("r_best_fintness: %f"%(r_best.fitness_))

        # for j in range(population.pop_size):
        #     print(str(population.target_pop_list[j].__str__()))
        #     print(population.target_pop_list[j].fitness_)
        print("finished!")


class MTaylorGPAlg(Alg):
    def __init__(self, max_generation, X_array, y_array, up_op_list=None, down_op_list=None, eval_op_list=None,
                 error_tolerance=None,
                 population=None, NewSparseRegressionFlag=False,
                 recursion_limit=300, repeat=1, originalTaylorGPGeneration=100, SR_method="gplearn",
                 mabPolicy="Greedy"):
        super().__init__(max_generation, up_op_list, down_op_list, eval_op_list, error_tolerance, population)
        self.NewSparseRegressionFlag = NewSparseRegressionFlag
        self.best_ind = None
        self.elapse_time = None
        self.best_fit = None
        self.mabPolicy = mabPolicy
        self.SR_method = SR_method
        self.X = X_array
        self.y = y_array
        self.originalTaylorGPGeneration = originalTaylorGPGeneration
        self.repeat = repeat
        self.recursion_limit = recursion_limit

    def run(self):
        flag1 = False
        flag2 = False

        np.set_printoptions(suppress=True)
        t = time.time()

        np_x = np.array(self.X)
        np_y = np.array(self.y)

        dataSets = np.hstack([np_x, np_y])

        # dataSets = self.ds

        # np_x = self.ds.get_np_x()
        # np_y = self.ds.get_np_y()
        # np_y = np_y.reshape([-1, 1])
        sys.setrecursionlimit(self.recursion_limit)

        # dataSets = np.column_stack(X_normalized, y_normalized)

        # print("维度： ", dataSets.shape[1] - 1)
        repeat = self.repeat
        totalGeneration = self.max_generation
        originalTaylorGPGeneration = self.originalTaylorGPGeneration
        Pop = self.population.pop_size  # 种群规模
        epsilons = [1e-5, 0.000001, 0.001, 0.01, 0.1, 0.2, 1, 2, 4, 10, 50, 100]
        # time_start1 = time.time()
        _init()
        for repeatNum in range(repeat):
            # time_start2 = time.time()
            SRC = subRegionCalculator(dataSets, originalTaylorGPGeneration, mabPolicy=self.mabPolicy,
                                      NewSparseRegressionFlag=self.NewSparseRegressionFlag)
            countAvailableParameters = SRC.CalCountofAvailableParameters(epsilons=epsilons, np_x=np_x)
            mabLoopNum = max(totalGeneration // originalTaylorGPGeneration // countAvailableParameters, 1)
            for epsilon in epsilons:
                print(epsilon)
                if flag1:
                    break
                if epsilon <= 1e-5:
                    SRC.PreDbscan(epsilon, noClusterFlag=True, clusterMethod="NOCLUSTER",
                                  data_x=np_x)  # 执行 OriginalTaylorGP
                elif not SRC.PreDbscan(epsilon, clusterMethod="DBSCAN", data_x=np_x):
                    print("聚类有问题")
                    continue
                SRC.firstMabFlag = True
                set_value('FIRST_EVOLUTION_FLAG', True)
                # 进行每轮数据集演化前执行
                for tryNum in range(mabLoopNum):
                    print("子块个数:"+str(len(SRC.subRegions)))
                    SRC.CalTops(repeatNum, Pop, SR_method=self.SR_method)
                    SRC.SubRegionPruning()
                    SRC.SparseRegression()
                    if SRC.bestLassoFitness < 1e-5:
                        print("Final Fitness", SRC.bestLassoFitness, " Selected SubRegon Index: ",
                              SRC.globalBestLassoCoef)
                        flag1 = True
                        break
                print("Temp Final Fitness", SRC.bestLassoFitness, " Selected SubRegon Index: ", SRC.globalBestLassoCoef)
            print("Final Fitness", SRC.bestLassoFitness, " Selected SubRegon Index: ", SRC.globalBestLassoCoef)
            self.best_fit = SRC.bestLassoFitness
        # if isinstance(SRC.globalBestLassoCoef, list):
        #     print(SRC.globalBestLassoCoef)
        #     print(SRC.tops)
        self.best_ind = SRC.tops[SRC.globalBestLassoCoef[0] - 1]
        # else:
        #     self.best_ind = SRC.tops[SRC.globalBestLassoCoef]
        print("best_ind" + str(self.best_ind[1][0]))
        dict_arr = SRC.dict_arr
        final_dict = {}
        # print(dict_arr)
        # for i in dict_arr:
        #     for key in i:
        #         if key not in final_dict:
        #             final_dict.update({key: i[key]})
        #         else:
        #             now_num = final_dict[key]
        #             now_num += i[key]
        #             final_dict.update({key: now_num})
        # print("final::::" + str(final_dict))
        self.elapse_time = time.time() - t
        str_eq = str(self.best_ind[1][0])
        str_eq = re.sub(r'x(\d{1})', r'x_\1', str_eq)
        self.best_ind = str_eq

    """
        for fileNum in range(19,20):
            fileName = "D:\PYcharm_program\Test_everything\Bench_0.15\BenchMark_" + str(fileNum) + ".tsv"
            dataSets = np.loadtxt(fileName,dtype=np.float,skiprows=1)
            TaylorGP2Master(dataSets)    
        """


class MTaylorKMeansAlg(Alg):
    def __init__(self, max_generation, ds, up_op_list=None, down_op_list=None, eval_op_list=None, error_tolerance=None,
                 population=None,
                 recursion_limit=300, repeat=1, originalTaylorGPGeneration=20, SR_method="gplearn", mabPolicy="Greedy",
                 recursionlimit=300):
        super().__init__(max_generation, up_op_list, down_op_list, eval_op_list, error_tolerance, population)
        self.recursionlimit = recursionlimit
        self.best_fit = None
        self.elapse_time = None
        self.mabPolicy = mabPolicy
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
        sys.setrecursionlimit(self.recursionlimit)
        t = time.time()
        dataSets = self.ds.get_np_ds()
        x = self.ds.get_np_x()
        average_fitness = 0
        totalGeneration = self.max_generation
        originalTaylorGPGen = 10
        Pop = self.population.pop_size  # 种群规模
        if dataSets.shape[1] - 1 == 1:
            clusters = [1]
        elif dataSets.shape[1] - 1 == 2:
            clusters = [2, 4]
        else:
            clusters = [1, 2, 4, 8, 16]
        time_start1 = time.time()
        for repeatNum in range(self.repeat):
            time_start2 = time.time()
            bestLassoFitness, globalBestLassoCoef, best_ind = Cal_fitness_Coef(dataSets, originalTaylorGPGen,
                                                                               totalGeneration,
                                                                               clusters, repeatNum, Pop, fileNum=1,
                                                                               np_x=x)
            average_fitness += bestLassoFitness
            time_end2 = time.time()
            print('current_time_cost', (time_end2 - time_start2) / 3600, 'hour')
        time_end1 = time.time()
        print('average_time_cost', (time_end1 - time_start1) / 3600 / self.repeat, 'hour')
        print('average_fitness = ', average_fitness / self.repeat)
        best_ind = trans_op2(best_ind)
        eval = SingleBingoEvaluator(data=self.ds, equation=best_ind)
        fit = eval.do()
        self.best_fit = fit
        self.elapse_time = time.time() - t
