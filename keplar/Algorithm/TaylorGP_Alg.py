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
import math


class TayloGPAlg(Alg):
    def __init__(self, generation, taylorGP_pre1, taylorGP_pre2, selector, creator, crossover, mutation, method_probs,
                 taylorsort, evaluator, max_generation, up_op_list, down_op_list, eval_op_list, error_tolerance,
                 population):
        super().__init__(max_generation, up_op_list, down_op_list, eval_op_list, error_tolerance, population)
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
        best_program = None
        best_program_fitness_ = None
        for gen in range(prior_generations, self.generations):
            top1Flag = False
            start_time = time()
            # if get_value('FIRST_EVOLUTION_FLAG') == False:
            if gen == 0:
                parents = population_input  # if第一轮，传的是空父母，如果第二轮，传的是非空父母，所以不用分开处理
                if parents != None:
                    self._programs.append(population_input)
                    best_program = parents[0]
                    best_program_fitness_ = parents[0]
                    continue
            else:  # 针对第二代演化父母都已经发生改变了，与是不是第一轮没有关系
                parents = self._programs[gen - 1]
                # 已经是排过序的了！！！
                # parents.sort(key=lambda x: x.raw_fitness_)
                # np.random.shuffle(parents)
                top1Flag = True
            n_jobs, n_programs, starts = _partition_estimators(
                self.population_size, self.n_jobs)
            seeds = random_state.randint(MAX_INT, size=self.population_size)

            population = Parallel(n_jobs=n_jobs,
                                  verbose=int(self.verbose > 1))(
                delayed(_parallel_evolve)(n_programs[i],
                                          parents,
                                          X,
                                          y,
                                          sample_weight,
                                          seeds[starts[i]:starts[i + 1]],
                                          params)
                for i in range(n_jobs))

            # Reduce, maintaining order across different n_jobs
            population = list(itertools.chain.from_iterable(population))
            if top1Flag:
                population.append(best_program_fitness_)
                population.append(best_program)
            fitness = [program.raw_fitness_ for program in population]
            length = [program.length_ for program in population]

            parsimony_coefficient = None
            if self.parsimony_coefficient == 'auto':
                parsimony_coefficient = (np.cov(length, fitness)[1, 0] /
                                         np.var(length))
            for program in population:
                program.fitness_ = program.fitness(parsimony_coefficient)
            fitness_ = [program.fitness_ for program in population]
            self._programs.append(population)

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
                best_program = population[np.argmax(fitness)]
                best_program_fitness_ = population[np.argmax(fitness_)]
            else:
                best_program = population[np.argmin(fitness)]
                best_program_fitness_ = population[np.argmin(fitness_)]

            self.run_details_['generation'].append(gen)
            self.run_details_['average_length'].append(np.mean(length))
            self.run_details_['average_fitness'].append(np.mean(fitness))
            self.run_details_['best_length'].append(best_program.length_)
            self.run_details_['best_fitness'].append(best_program.raw_fitness_)
            oob_fitness = np.nan
            if self.max_samples < 1.0:
                oob_fitness = best_program.oob_fitness_
            self.run_details_['best_oob_fitness'].append(oob_fitness)
            generation_time = time() - start_time
            self.run_details_['generation_time'].append(generation_time)

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
