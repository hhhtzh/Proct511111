import itertools
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
from keplar.translator.translator import trans_taylor_program, taylor_trans_population
# from TaylorGP.src.taylorGP.genetic import BaseSymbolic
from TaylorGP.src.taylorGP.fitness import _mean_square_error, _weighted_spearman, _log_loss, _mean_absolute_error, \
    _Fitness
from TaylorGP.src.taylorGP._program import _Program, print_program

import math
from gplearn.utils import _partition_estimators, check_random_state
from joblib import Parallel, delayed  # 自动创建进程池执行并行化操作
from keplar.population.population import Population

MAX_INT = np.iinfo(np.int32).max


def _parallel_evolve(n_programs, population, X, y, sample_weight, seeds, params, crossover, mutation):
    """Private function used to build a batch of programs within a job."""
    n_samples, n_features = X.shape
    # Unpack parameters
    tournament_size = params['tournament_size']
    function_set = params['function_set']
    arities = params['arities']
    init_depth = params['init_depth']
    init_method = params['init_method']
    const_range = params['const_range']
    metric = params['_metric']
    transformer = params['_transformer']
    parsimony_coefficient = params['parsimony_coefficient']
    method_probs = params['method_probs']
    p_point_replace = params['p_point_replace']
    max_samples = params['max_samples']
    feature_names = params['feature_names']
    selected_space = params['selected_space']
    qualified_list = params['qualified_list']  # 合格判别标准
    eq_write = params['eq_write']  # 用于将所有生成的公式写入eq_write文件

    max_samples = int(max_samples * n_samples)

    def _tournament():
        """Find the fittest individual from a sub-population."""
        contenders = random_state.randint(0, len(parents), tournament_size)
        fitness = [parents[p].fitness_ for p in contenders]
        if metric.greater_is_better:
            parent_index = contenders[np.argmax(fitness)]
        else:
            parent_index = contenders[np.argmin(fitness)]
        return parents[parent_index], parent_index

    # Build programs
    programs = []
    parents = population.target_pop_list

    for i in range(n_programs):
        random_state = check_random_state(i)

        if parents is None:
            program = None
            genome = None
        else:
            method = random_state.uniform()
            parent, parent_index = _tournament()
            if method < method_probs[0]:

                # crossover
                donor, donor_index = _tournament()

                crossover.get_value(random_state, parent, donor.program, i)
                program, removed, remains = crossover.do(population)
                # program =None

                genome = {'method': 'Crossover',
                          'parent_idx': parent_index,
                          # 'parent_nodes': removed,
                          'donor_idx': donor_index
                          # 'donor_nodes': remains}
                          }
            elif method < method_probs[1]:
                # subtree_mutation
                mutation.get_value(1, random_state,  parent, i)
                program, removed, _ = mutation.do(population)
                # program =None


# <<<<<<< HEAD
                # program, removed, _ = parent.subtree_mutation(random_state)
                genome = {'method': 'Subtree Mutation',
                          'parent_idx': parent_index,
                          'parent_nodes': removed}
            elif method < method_probs[2]:
                # hoist_mutation
                mutation.get_value(2, random_state,  parent, i)
                program, removed = mutation.do(population)

                # program, removed = parent.hoist_mutation(random_state)
                genome = {'method': 'Hoist Mutation',
                          'parent_idx': parent_index,
                          'parent_nodes': removed}
            elif method < method_probs[3]:
                program = None
                genome = None

            else:
                # program = None
                # genome = None
                mutation.get_value(4, random_state,  parent, i)
                program = mutation.do(population)

                # reproduction
                # program = parent.reproduce()
                genome = {'method': 'Reproduction',
                          'parent_idx': parent_index,
                          'parent_nodes': []}

        program = _Program(function_set=function_set,
                           arities=arities,
                           init_depth=init_depth,
                           init_method=init_method,
                           n_features=n_features,
                           metric=metric,
                           transformer=transformer,
                           const_range=const_range,
                           p_point_replace=p_point_replace,
                           parsimony_coefficient=parsimony_coefficient,
                           feature_names=feature_names,
                           random_state=random_state,
                           program=program,
                           selected_space=selected_space,
                           qualified_list=qualified_list,
                           X=X,
                           eq_write=eq_write)

        # program.parents = genome

        # Draw samples, using sample weights, and then fit
        if sample_weight is None:
            curr_sample_weight = np.ones((n_samples,))
        else:
            curr_sample_weight = sample_weight.copy()
        oob_sample_weight = curr_sample_weight.copy()

        indices, not_indices = program.get_all_indices(n_samples,
                                                       max_samples,
                                                       random_state)

        curr_sample_weight[not_indices] = 0
        oob_sample_weight[indices] = 0

        program.raw_fitness_ = program.raw_fitness(X, y, curr_sample_weight)
        if math.isnan(program.raw_fitness_) or math.isinf(program.raw_fitness_) or program.length_ > 500:
            i -= 1
            continue
        if max_samples < n_samples:
            # Calculate OOB fitness
            program.oob_fitness_ = program.raw_fitness(X, y, oob_sample_weight)

        programs.append(program)

    return programs


class TayloGPAlg(Alg):
    def __init__(self, generation, taylorGP_pre1, taylorGP_pre2, selector, creator, crossover, mutation, method_probs,
                 taylorsort, evaluator):
        # super().__init__(max_generation, up_op_list, down_op_list, eval_op_list, error_tolerance, population)
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
        self.population_size = 128
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
        self.n_jobs = 1
        self._programs = []
        self.low_memory = True
        self._metric = None

        self.best_fit = None
        self.elapse_time = None

    def _verbose_reporter(self, run_details=None):
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
            gen = run_details['generation'][-1]
            # generation_time = run_details['generation_time'][-1]
            # remaining_time = (self.generations - gen - 1) * generation_time
# <<<<<<< HEAD
            remaining_time = 0
            if remaining_time > 60:
                remaining_time = '{0:.2f}m'.format(remaining_time / 60.0)
            else:
                remaining_time = '{0:.2f}s'.format(remaining_time)
# =======
            # if remaining_time > 60:
            #     remaining_time = '{0:.2f}m'.format(remaining_time / 60.0)
            # else:
            #     remaining_time = '{0:.2f}s'.format(remaining_time)
            remaining_time = '{0:.2f}s'.format(0.0)
# >>>>>>> c0ed278302bd149db82e0ea24012f794506aed24

            oob_fitness = 'N/A'
            line_format = '{:4d} {:8.2f} {:16g} {:8d} {:16g} {:>16} {:>10}'
            if self.max_samples < 1.0:
                oob_fitness = run_details['best_oob_fitness'][-1]
                line_format = '{:4d} {:8.2f} {:16g} {:8d} {:16g} {:16g} {:>10}'

            print(line_format.format(run_details['generation'][-1],
                                     run_details['average_length'][-1],
                                     run_details['average_fitness'][-1],
                                     run_details['best_length'][-1],
                                     run_details['best_fitness'][-1],
                                     oob_fitness,
                                     # <<<<<<< HEAD
                                     #                                      remaining_time))

                                     #     def select_by_crowding_distance(self,population,front,reminder):
                                     # =======
                                     remaining_time
                                     ))

    def select_by_crowding_distance(self, population, front, reminder):
        # >>>>>>> c0ed278302bd149db82e0ea24012f794506aed24
        cur_population = copy.deepcopy([population[i] for i in front])
        cur_population.sort(key=lambda x: x.raw_fitness_)
        sorted1 = copy.deepcopy(cur_population)
        cur_population.sort(key=lambda x: x.length_)
        sorted2 = cur_population
        distance = [0 for i in range(0, len(front))]
        distance[0] = 4444444444444444
        distance[len(front) - 1] = 4444444444444444
        fitness_ = [sorted1[i].raw_fitness_ for i in range(
            len(cur_population))]
        length_ = [sorted2[i].length_ for i in range(len(cur_population))]
        maxFit, minFit, maxLen, minLen = max(fitness_), min(
            fitness_), max(length_), min(length_)
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
        t = time.time()

        X, Y, qualified_list = self.taylorGP_pre1.do()
        self.taylorGP_pre2.get_value(X, Y, qualified_list)
# <<<<<<< HEAD
        X, y, params, population_size, seeds, qualified_list, function_set, n_features, sample_weight = self.taylorGP_pre2.do()
# =======
#         X, y, params, population_size, seeds, qualified_list, function_set, n_features = self.taylorGP_pre2.do()
# >>>>>>> c0ed278302bd149db82e0ea24012f794506aed24
#         parents = None

        n_samples, n_features = X.shape
        max_samples = params['max_samples']
        self._metric = params['_metric']

        max_samples = int(max_samples * n_samples)
        population_input = None

        # seeds = random_state.randint(MAX_INT, size=self.population_size)
        best_program = None
        best_program_fitness_ = None
        population = Population(self.population_size)
        population.target_pop_list = None
        for gen in range(self.generation):
            top1Flag = False
            # start_time = time()
            # if get_value('FIRST_EVOLUTION_FLAG') == False:
            if gen == 0:
                parents = population_input  # if第一轮，传的是空父母，如果第二轮，传的是非空父母，所以不用分开处理
                if parents != None:
                    self._programs.append(population_input)
                    best_program = parents[0]
                    best_program_fitness_ = parents[0]
                    continue
                else:
                    self._programs.append(None)
# <<<<<<< HEAD
                self._verbose_reporter()
#             else:#针对第二代演化父母都已经发生改变了，与是不是第一轮没有关系
# =======
            else:  # 针对第二代演化父母都已经发生改变了，与是不是第一轮没有关系
                # >>>>>>> c0ed278302bd149db82e0ea24012f794506aed24
                parents = self._programs[gen - 1]
                # 已经是排过序的了！！！
                # parents.sort(key=lambda x: x.raw_fitness_)
                # np.random.shuffle(parents)
                top1Flag = True
            n_jobs, n_programs, starts = _partition_estimators(
                self.population_size, self.n_jobs)
            random_state = check_random_state(None)

            seeds = random_state.randint(MAX_INT, size=self.population_size)

            population.target_pop_list = Parallel(n_jobs=n_jobs,
                                                  verbose=int(self.verbose > 1))(
                delayed(_parallel_evolve)(n_programs[i],
                                          population,
                                          X,
                                          y,
                                          sample_weight,
                                          seeds[starts[i]:starts[i + 1]],
                                          params,
                                          self.crossover,
                                          self.mutation)
                for i in range(n_jobs))
            population.target_pop_list = list(itertools.chain.from_iterable(population.target_pop_list))
            if top1Flag:
                population.target_pop_list.append(best_program_fitness_)
                population.target_pop_list.append(best_program)
            fitness = [program.raw_fitness_ for program in population.target_pop_list]
            length = [program.length_ for program in population.target_pop_list]

            parsimony_coefficient = None
            if self.parsimony_coefficient == 'auto':
                parsimony_coefficient = (np.cov(length, fitness)[1, 0] /
                                         np.var(length))
            for program in population.target_pop_list:
                program.fitness_ = program.fitness(parsimony_coefficient)
            fitness_ = [program.fitness_ for program in population.target_pop_list]
            self._programs.append(population.target_pop_list)

            # Remove old programs that didn't make it into the new population.

            if not self.low_memory:
                for old_gen in np.arange(gen, 0, -1):
                    indices = []
                    for program in self._programs[old_gen]:
                        if program is not None and program.parents is not None:
                            for idx in program.parents:
                                if 'idx' in idx:
                                    indices.append(program.parents[idx])
                    indices = set(indices)
                    for idx in range(self.population_size):
                        if idx not in indices:
                            self._programs[old_gen - 1][idx] = None
            elif gen > 0:
                # Remove old generations
                self._programs[gen - 1] = None

            # Record run details
            if self._metric.greater_is_better:
                best_program = population.target_pop_list[np.argmax(fitness)]
                best_program_fitness_ = population.target_pop_list[np.argmax(fitness_)]
            else:
                best_program = population.target_pop_list[np.argmin(fitness)]
                best_program_fitness_ = population.target_pop_list[np.argmin(fitness_)]

            self.run_details_['generation'].append(gen)
            self.run_details_['average_length'].append(np.mean(length))
            self.run_details_['average_fitness'].append(np.mean(fitness))
            self.run_details_['best_length'].append(best_program.length_)
            self.run_details_['best_fitness'].append(best_program.raw_fitness_)
            oob_fitness = np.nan
            if self.max_samples < 1.0:
                oob_fitness = best_program.oob_fitness_
            self.run_details_['best_oob_fitness'].append(oob_fitness)
            # generation_time = time() - start_time
            # self.run_details_['generation_time'].append(generation_time)

            if self.verbose:
                self._verbose_reporter(self.run_details_)

            # Check for early stopping
            if self._metric.greater_is_better:
                best_fitness = fitness[np.argmax(fitness)]
                if best_fitness >= self.stopping_criteria:
                    break
            else:
                best_fitness = fitness[np.argmin(fitness)]
                if best_fitness <= self.stopping_criteria:
                    break



        self.elapse_time = time.time() - t
