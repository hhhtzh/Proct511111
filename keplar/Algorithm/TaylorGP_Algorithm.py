import random
import sys
import time
import argparse
import copy

import numpy as np
from numpy import shape
from sympy import symbols

from TaylorGP.src.taylorGP._global import set_value, _init
from TaylorGP.src.taylorGP.subRegionCalculator import subRegionCalculator
from keplar.Algorithm.Alg import Alg
from TaylorGP.src.taylorGP.utils import check_random_state
from keplar.translator.translator import trans_taylor_program,taylor_trans_population
# from TaylorGP.src.taylorGP.genetic import BaseSymbolic
from TaylorGP.src.taylorGP.fitness import _mean_square_error, _weighted_spearman, _log_loss, _mean_absolute_error, \
    _Fitness
import math




class TayloGPAlg(Alg):
    def __init__(self, generation,taylorGP_pre1,taylorGP_pre2, selector,creator,crossover,mutation,method_probs,taylorsort,evaluator):
        self.generation=generation
        self.taylorGP_pre1=taylorGP_pre1
        self.taylorGP_pre2=taylorGP_pre2
        self.selector=selector
        self.creator=creator
        self.crossover=crossover
        self.mutation=mutation
        self.method_probs=method_probs
        self.evaluator=evaluator
        self.taylorsort=taylorsort
        self.parsimony_coefficient=0.001
        self.population_size = 1000
        self.run_details_ = {'generation': [],
                                 'average_length': [],
                                 'average_fitness': [],
                                 'best_length': [],
                                 'best_fitness': [],
                                 'best_oob_fitness': [],
                                 'generation_time': []}
        self.max_samples =1.0
        self.verbose = 1
        self.stopping_criteria = 0.0
        self.sample_weight = None
        # self.params= params
        # self.X =X
        # self.y =y 


    
    def print_details(self, run_details=None ,i =None):
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

    def select_by_crowding_distance(self,population,front,reminder):
        cur_population = copy.deepcopy([population[i] for i in front])
        cur_population.sort(key=lambda x: x.raw_fitness_)
        sorted1 = copy.deepcopy(cur_population)
        cur_population.sort(key=lambda x: x.length_)
        sorted2 = cur_population
        distance = [0 for i in range(0, len(front))]
        distance[0] = 4444444444444444
        distance[len(front) - 1] = 4444444444444444
        fitness_ = [sorted1[i].raw_fitness_ for i in range(len(cur_population))]
        length_ =  [sorted2[i].length_ for i in range(len(cur_population))]
        maxFit,minFit,maxLen,minLen = max(fitness_),min(fitness_),max(length_),min(length_)
        #第k个个体的距离就是front[k]的距离----dis[k]==front[k]
        for k in range(1, len(front) - 1):
            distance[k] = distance[k] + (fitness_[k + 1] - fitness_[k - 1]) / (
                        maxFit - minFit+0.01)
        for k in range(1, len(front) - 1):
            distance[k] = distance[k] + (length_[k + 1] - length_[k - 1]) / (
                        maxLen - minLen+0.01)
        index_ = sorted(range(len(distance)),key=lambda k:distance[k])
        index_.reverse()
        reminderPop = [cur_population[i] for i in index_][:reminder]
        return reminderPop

    def run(self):
        X, Y, qualified_list = self.taylorGP_pre1.do()
        self.taylorGP_pre2.get_value(X, Y, qualified_list)
        X,y,params,population_size,seeds,qualified_list,function_set,n_features= self.taylorGP_pre2.do()
        parents = None

        n_samples, n_features = X.shape
        max_samples = params['max_samples']
        max_samples = int(max_samples * n_samples)





        
        for gen in range(self.generation):
            programs = []
            if gen==0:
                population,sample_weight = self.creator.do()
                self.print_details()
            else:
                # print(population.pop_size)
                # for j in range(population.pop_size):
                j=0
                while j != population.pop_size:
                    random_state = check_random_state(j)
                    method = random_state.uniform()
                    # program  = trans_taylor_program(population.target_pop_list[j])
                    self.selector.get_value(random_state,tournament_size=50)
                    pop_parent,pop_best_index = self.selector.do(parents)

                    if method < self.method_probs[0]:
                        # print("ttt")
                        # print(population.target_pop_list[j].get_expression())

                        self.selector.get_value(random_state,tournament_size=50)
                        pop_honor,honor_best_index = self.selector.do(parents)
                        # print("how")
                        self.crossover.get_value(random_state,pop_parent,pop_honor.program,j)
                        population = self.crossover.do(population)
                        # print("how")
                        
                        # population.target_pop_list[]

                    elif method < self.method_probs[1]:
                        # print("rrrr")
       
                        self.mutation.get_value(1, random_state,  pop_parent, j)
                        population =self.mutation.do(population)
                        # print("how2")


                    elif method < self.method_probs[2]:
                        self.mutation.get_value(2, random_state, pop_parent, j)
                        population =self.mutation.do(population)
                    
                    elif method < self.method_probs[3]:
                        self.mutation.get_value(3, random_state,  pop_parent, j)
                        population = self.mutation.do(population)

                    else:
                        self.mutation.get_value(4, random_state,  pop_parent, j)
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
                    program =population.target_pop_list[j]

                    
                    indices, not_indices = program.get_all_indices(n_samples,
                                                                max_samples,
                                                                random_state)

                    curr_sample_weight[not_indices] = 0
                    oob_sample_weight[indices] = 0


                    program.raw_fitness_ = program.raw_fitness(X, y, curr_sample_weight)
          
                    if math.isnan(program.raw_fitness_) or math.isinf(program.raw_fitness_) or program.length_ >500:
                        # i =i- 1
                        # i -= 1
                        # idx = i
                        # print(i)
                        # n_pop += 1
                        continue
                        # pass
                    program.fitness_ = program.fitness(self.parsimony_coefficient)
                    # print("test111")
                    # print(program.fitness_)

                    # if max_samples < n_samples:
                    #     # Calculate OOB fitness
                    #     program.oob_fitness_ = program.raw_fitness(self.X, self.y, oob_sample_weight)
                    
                    # if idx == i:
                    #     print("????")
                    # population.target_append(program)



                    j+=1



                    
                    # print("5555")
                    # print(j)
                    # print(population.target_pop_list[j].get_expression())

                    # self.evaluator.get_value()
                    # print(population.target_pop_list[j].fitness_)

                    # population
                population = self.evaluator.do(population)

                if parents is not None:
                    pass
                    # for i in range(parents.pop_size):
                    #     population.target_append(parents.target_pop_list[i])

                temp_index = self.taylorsort.do(population)
                # print("kkkk")
                # print(len(temp_index))
                # print(temp_index)
                
                
                temp_popSize = 0
                population_index = []
                reminder_subPopulation = []
                for subPop in temp_index:
                    pre_temp_popSize = temp_popSize
                    temp_popSize += len(subPop)
                    if temp_popSize >self.population_size:
                        reminder = self.population_size-pre_temp_popSize
                        # print("temp_popSize: ",temp_popSize,"reminder: ",reminder)
                        # print(len())
                        # print(self.select_by_crowding_distance(population.target_pop_list,subPop,reminder))
                        reminder_pop = self.select_by_crowding_distance(population.target_pop_list,subPop,reminder)
                        for h in range(len(reminder_pop)):
                            reminder_subPopulation.append(reminder_pop[h])

                        # reminder_subPopulation.target_append(self.select_by_crowding_distance(population.target_pop_list,subPop,reminder))
                        # print("reminder_subPopulation: ",reminder_subPopulation)
                        break
                    else:
                        population_index.extend(subPop)
                
                population.target_pop_list = [population.target_pop_list[i] for i in population_index]
                if reminder_subPopulation !=[]:
                    # population.target_append(reminder_subPopulation)
                    pass

                fitness = [program.raw_fitness_ for program in population.target_pop_list]
                length = [program.length_ for program in population.target_pop_list]

                parsimony_coefficient = None
                if self.parsimony_coefficient == 'auto':
                    parsimony_coefficient = (np.cov(length, fitness)[1, 0] /
                                            np.var(length))
                for program in population.target_pop_list:
                    program.fitness_ = program.fitness(parsimony_coefficient)
                fitness_ = [program.fitness_ for program in population.target_pop_list]

                if self.selector.greater_is_better:
                    best_program = population.target_pop_list[np.argmax(fitness)]#按惩罚项的fitness排序
                    best_program_fitness_ = population.target_pop_list[np.argmax(fitness_)]
                else:
                    best_program = population.target_pop_list[np.argmin(fitness)]
                    best_program_fitness_ = population.target_pop_list[np.argmin(fitness_)]

                self.run_details_['generation'].append(gen)
                self.run_details_['average_length'].append(np.mean(length))
                self.run_details_['average_fitness'].append(np.mean(fitness_))
                self.run_details_['best_length'].append(best_program.length_)
                self.run_details_['best_fitness'].append(best_program.fitness_)
                oob_fitness = np.nan
                if self.max_samples < 1.0:
                    oob_fitness = best_program.oob_fitness_
                self.run_details_['best_oob_fitness'].append(oob_fitness)

                if self.verbose:
                    # print("uuuu")
                    # print(gen)
                    print(self.run_details_['generation'][0])
                    self.print_details(self.run_details_,gen-1)

                # Check for early stopping
                if self.selector.greater_is_better:
                    best_fitness = fitness[np.argmax(fitness_)]
                    if best_fitness >= self.stopping_criteria:
                        break
                else:
                    best_fitness = fitness[np.argmin(fitness_)]
                    if best_fitness <= self.stopping_criteria:
                        break


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
