from keplar.operator.operator import Operator
import signal
import numpy as np
from time import time, sleep
from sympy import *

from TaylorGP.src.taylorGP.genetic import alarm_handler, MAX_INT, _parallel_evolve, BaseEstimator, BaseSymbolic
from TaylorGP.src.taylorGP.calTaylor import Metrics, Metrics2
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error  # 均方误差

from TaylorGP.src.taylorGP.utils import check_random_state
from sklearn.utils.validation import check_X_y, check_array
from sklearn.base import RegressorMixin, TransformerMixin, ClassifierMixin
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils import compute_sample_weight
from TaylorGP.src.taylorGP.functions import _function_map, _Function, sig1 as sigmoid
from TaylorGP.src.taylorGP.fitness import _fitness_map, _Fitness
from TaylorGP.src.taylorGP.judge_bound import select_space, cal_spacebound
from warnings import warn
from TaylorGP.src.taylorGP.utils import _partition_estimators
from joblib import Parallel, delayed  # 自动创建进程池执行并行化操作
import itertools
from sklearn.base import RegressorMixin, TransformerMixin, ClassifierMixin
from sklearn.base import BaseEstimator
# @abc.abstractmethod装饰器后严格控制子类必须实现这个方法
from abc import ABCMeta, abstractmethod
from TaylorGP.src.taylorGP.calTaylor import Metrics, Metrics2

# from sklearn.base.BaseEstimator

# from sklearn.base import BaseEstimator


class TimeOutException(Exception):
    pass


class TaylorGP_Pre1(Operator):
    def __init__(self, X, y):
        super().__init__()

        self.X = X
        self.y = y

        self.max_time = 60

    def do(self, population=None):
        # return super().do(population)
        signal.signal(signal.SIGALRM, alarm_handler)
        # signal.alarm(MAXTIME)
        signal.alarm(self.max_time)  # maximum time, defined above
        '''

        p = Thread(target=self.thread_test)
        p.start()
        '''

        try:
            # np.expand_dims(y,axis=1)
            y = self.y[:, np.newaxis]
            # y= y.reshape(-1)
            X_Y = np.concatenate((self.X, y), axis=1)
            print(X_Y.shape)

            # X_Y = np.array(X)[1:].astype(np.float)
            x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21,\
                x22, x23, x24, x25, x26, x27, x28, x29, x30, x31, x32, x33, x34, x35, x36, x37, x38, x39, x40, x41, x42,\
                x43, x44, x45, x46, x47, x48, x49,\
                x50, x51, x52, x53, x54, x55, x56, x57, x58, x59, x60, x61, x62, x63, x64, x65, x66, x67, x68, x69, x70,\
                x71, x72, x73, x74, x75, x76, x77, x78, x79,\
                x80, x81, x82, x83, x84, x85, x86, x87, x88, x89, x90, x91, x92, x93, x94, x95, x96, x97, x98, x99, x100 = symbols(
                    "x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21,\
                  x22, x23, x24, x25, x26, x27, x28, x29, x30, x31, x32, x33, x34, x35, x36, x37, x38, x39, x40, x41, x42,\
                  x43, x44, x45, x46, x47, x48, x49,\
                  x50, x51, x52, x53, x54, x55, x56, x57, x58, x59, x60, x61, x62, x63, x64, x65, x66, x67, x68, x69, x70,\
                  x71, x72, x73, x74, x75, x76, x77, x78, x79,\
                  x80, x81, x82, x83, x84, x85, x86, x87, x88, x89, x90, x91, x92, x93, x94, x95, x96, x97, x98, x99, x100 ")
            _x = [x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21,
                  x22, x23, x24, x25, x26, x27, x28, x29, x30, x31, x32, x33, x34, x35, x36, x37, x38, x39, x40, x41, x42,
                  x43, x44, x45, x46, x47, x48, x49,
                  x50, x51, x52, x53, x54, x55, x56, x57, x58, x59, x60, x61, x62, x63, x64, x65, x66, x67, x68, x69, x70,
                  x71, x72, x73, x74, x75, x76, x77, x78, x79,
                  x80, x81, x82, x83, x84, x85, x86, x87, x88, x89, x90, x91, x92, x93, x94, x95, x96, x97, x98, x99, x100]
            self._x = _x
            average_fitness = 0
            repeat = 1
            time_start1 = time()
            time_start2 = time()
            loopNum = 0
            Metric = []
            while True:
                print(X_Y.shape)
                '''
                for i in range(X_Y.shape[0]):
                    for j in range(i + 1, X_Y.shape[0]):
                        if ((X_Y[i] == X_Y[j]).all()):
                            print("data is same in :" + str(i) + " " + str(j))            
                '''
                # 使用去重后的数据计算Taylor展开式
                metric = Metrics(
                    varNum=self.X.shape[1], dataSet=np.unique(X_Y, axis=0))
                loopNum += 1
                Metric.append(metric)
                if metric.nmse > 10000:
                    print("use Linear regression")
                    break
                if loopNum == 6 and self.X.shape[1] <= 2:
                    break
                elif loopNum == 5 and (self.X.shape[1] > 2 and self.X.shape[1] <= 3):
                    break
                elif loopNum == 4 and (self.X.shape[1] > 3 and self.X.shape[1] <= 4):
                    break
                elif loopNum == 3 and (self.X.shape[1] > 4 and self.X.shape[1] <= 5):
                    break
                elif loopNum == 2 and (self.X.shape[1] > 5 and self.X.shape[1] <= 6):
                    break
                elif loopNum == 1 and (self.X.shape[1] > 6):
                    break
            Metric.sort(key=lambda x: x.nmse)
            metric = Metric[0]
            print('NMSE of polynomial and lower order polynomial after sorting:',
                  metric.nmse, metric.low_nmse)
            if metric.nmse < 0.01:
                metric.nihe_flag = True
            else:
                print("call  Linear regression to change nmse and f_taylor")
                lr_est = LinearRegression().fit(self.X, y)
                print('coef: ', lr_est.coef_)
                print('intercept: ', lr_est.intercept_)
                lr_nmse = mean_squared_error(
                    lr_est.predict(self.X), y, squared=False)
                if lr_nmse < metric.nmse:
                    metric.nmse = lr_nmse
                    metric.low_nmse = lr_nmse
                    f = str(lr_est.intercept_[0])
                    for i in range(self.X.shape[1]):
                        if lr_est.coef_[0][i] >= 0:
                            f += '+' + str(lr_est.coef_[0][i]) + '*x' + str(i)
                        else:
                            f += str(lr_est.coef_[0][i]) + '*x' + str(i)
                    print("f_lr and nmse_lr"+f + "  "+str(lr_nmse))
                    '''
                    fitness = mean_squared_error(lr_est.predict(test_X), test_y, squared=False)  # RMSE
                    print('LR_predict_fitness: ', fitness)                
                    '''
                    metric.f_taylor = sympify(f)
                    metric.f_low_taylor = sympify(f)
                metric.bias = 0.
                if lr_nmse < 0.1:
                    print('Fitting failed')
            time_end2 = time()
            print('Pretreatment_time_cost',
                  (time_end2 - time_start2) / 3600, 'hour')
            self.global_fitness, self.sympy_global_best = metric.low_nmse, metric.f_low_taylor
            if metric.judge_Low_polynomial():
                self.global_fitness, self.sympy_global_best = metric.low_nmse, metric.f_low_taylor
                '''
                elif metric.nihe_flag and (metric.judge_additi_separability() or metric.judge_multi_separability() ):
                    self.global_fitness,self.sympy_global_best = self.CalTaylorFeatures(metric.f_taylor,_x[:X.shape[1]],X,y,self.population_size,11111)
                '''
            else:
                qualified_list = []
                qualified_list.extend(
                    [metric.judge_Bound(),  # ok
                     metric.f_low_taylor,
                     metric.low_nmse,
                     metric.bias,
                     metric.judge_parity(),
                     metric.judge_monotonicity()])
                print(qualified_list)

                return self.X, metric.change_Y(y), qualified_list

        except TimeOutException:

            print("TimeOutException catched in fit()")

# class TaylorGP_pre2(Operator,BaseEstimator,RegressorMixin):


class TaylorGP_pre2(Operator, BaseSymbolic, RegressorMixin):
    def __init__(self, X, Y, qualified_list):
        super().__init__()
        self.X = X
        self.Y = Y
        self.qualified_list = qualified_list
        self.random_state = None
        self.sample_weight = None
        self.hall_of_fame = None
        self.population_size = 1000
        self.n_components = None
        self.function_set = ['add', 'sub', 'mul',
                             'div', 'sin', 'cos', 'log', 'exp', 'sqrt']
        self.metric = 'rmse'
        self.p_crossover = 0.9
        self.p_subtree_mutation = 0.01
        self.p_hoist_mutation = 0.01
        self.p_point_mutation = 0.01
        self.p_point_replace = 0.05

        self.init_method = 'half and half'
        self.const_range = (-1., 1.)
        self.init_depth = (2, 6)
        self.feature_names = None
        self.transformer = None
        # self._metric= None
        self.warm_start = False
        self.generations = 20
        self.verbose = 0
        self.n_jobs = 1
        self.tournament_size = 20
        self.parsimony_coefficient = 0.01
        self.max_samples = 1.0
        self.low_memory = True
        self.stopping_criteria = 0.0
        # self.feature_names

        # self.p_crossover = 0.9
        # self.p_subtree_mutation=0.01,
        # self.p_hoist_mutation=0.01,
        # self.p_point_mutation=0.01,
        # self.p_point_replace=0.05,

    def get_value(self,X,Y,qualified_list):
        self.X = X
        self.Y = Y
        self.qualified_list = qualified_list

    def do(self, population=None):
        # return super().do(population)
        low_bound, high_bound, var_bound = self.qualified_list[0][0][
            0], self.qualified_list[0][0][1], self.qualified_list[0][1]
        random_state = check_random_state(self.random_state)

        # Check arrays
        if self.sample_weight is not None:
            self.sample_weight = check_array(
                self.sample_weight, ensure_2d=False)

        if isinstance(self, ClassifierMixin):
            X, y = check_X_y(X, y, y_numeric=False)
            check_classification_targets(y)

            if self.class_weight:
                if self.sample_weight is None:
                    self.sample_weight = 1.
                # modify the sample weights with the corresponding class weight
                self.sample_weight = (self.sample_weight *
                                      compute_sample_weight(self.class_weight, y))

            self.classes_, y = np.unique(y, return_inverse=True)
            n_trim_classes = np.count_nonzero(
                np.bincount(y, self.sample_weight))
            if n_trim_classes != 2:
                raise ValueError("y contains %d class after sample_weight "
                                 "trimmed classes with zero weights, while 2 "
                                 "classes are required."
                                 % n_trim_classes)
            self.n_classes_ = len(self.classes_)

        else:
            X, y = check_X_y(self.X, self.Y, y_numeric=True)

        _, self.n_features_ = X.shape

        hall_of_fame = self.hall_of_fame
        if hall_of_fame is None:
            hall_of_fame = self.population_size
        if hall_of_fame > self.population_size or hall_of_fame < 1:
            raise ValueError('hall_of_fame (%d) must be less than or equal to '
                             'population_size (%d).' % (self.hall_of_fame,
                                                        self.population_size))
        n_components = self.n_components
        if n_components is None:
            n_components = hall_of_fame
        if n_components > hall_of_fame or n_components < 1:
            raise ValueError('n_components (%d) must be less than or equal to '
                             'hall_of_fame (%d).' % (self.n_components,
                                                     self.hall_of_fame))

        self._function_set = []
        for function in self.function_set:
            if isinstance(function, str):
                if function not in _function_map:
                    raise ValueError('invalid function name %s found in '
                                     '`function_set`.' % function)
                self._function_set.append(_function_map[function])
            elif isinstance(function, _Function):
                self._function_set.append(function)
            else:
                raise ValueError('invalid type %s found in `function_set`.'
                                 % type(function))
        if not self._function_set:
            raise ValueError('No valid functions found in `function_set`.')

        # For point-mutation to find a compatible replacement node
        self._arities = {}
        for function in self._function_set:  # 以函数集的arity个数对函数集划分
            arity = function.arity
            self._arities[arity] = self._arities.get(arity, [])
            self._arities[arity].append(function)

        if isinstance(self.metric, _Fitness):
            self._metric = self.metric
        elif isinstance(self, RegressorMixin):
            if self.metric not in ('mean absolute error', 'mse', 'rmse',
                                   'pearson', 'spearman'):
                raise ValueError('Unsupported metric: %s' % self.metric)
            self._metric = _fitness_map[self.metric]
        elif isinstance(self, ClassifierMixin):
            if self.metric != 'log loss':
                raise ValueError('Unsupported metric: %s' % self.metric)
            self._metric = _fitness_map[self.metric]
        elif isinstance(self, TransformerMixin):
            if self.metric not in ('pearson', 'spearman'):
                raise ValueError('Unsupported metric: %s' % self.metric)
            self._metric = _fitness_map[self.metric]

        self._method_probs = np.array([self.p_crossover,
                                       self.p_subtree_mutation,
                                       self.p_hoist_mutation,
                                       self.p_point_mutation])

        self._method_probs = np.cumsum(self._method_probs)

        if self._method_probs[-1] > 1:
            raise ValueError('The sum of p_crossover, p_subtree_mutation, '
                             'p_hoist_mutation and p_point_mutation should '
                             'total to 1.0 or less.')

        if self.init_method not in ('half and half', 'grow', 'full'):
            raise ValueError('Valid program initializations methods include '
                             '"grow", "full" and "half and half". Given %s.'
                             % self.init_method)

        if not ((isinstance(self.const_range, tuple) and
                len(self.const_range) == 2) or self.const_range is None):
            raise ValueError('const_range should be a tuple with length two, '
                             'or None.')

        if (not isinstance(self.init_depth, tuple) or
                len(self.init_depth) != 2):
            raise ValueError('init_depth should be a tuple with length two.')
        if self.init_depth[0] > self.init_depth[1]:
            raise ValueError('init_depth should be in increasing numerical '
                             'order: (min_depth, max_depth).')

        if self.feature_names is not None:
            if self.n_features_ != len(self.feature_names):
                raise ValueError('The supplied `feature_names` has different '
                                 'length to n_features. Expected %d, got %d.'
                                 % (self.n_features_, len(self.feature_names)))
            for feature_name in self.feature_names:
                if not isinstance(feature_name, str):
                    raise ValueError('invalid type %s found in '
                                     '`feature_names`.' % type(feature_name))

        if self.transformer is not None:
            if isinstance(self.transformer, _Function):
                self._transformer = self.transformer
            elif self.transformer == 'sigmoid':
                self._transformer = sigmoid
            else:
                raise ValueError('Invalid `transformer`. Expected either '
                                 '"sigmoid" or _Function object, got %s' %
                                 type(self.transformer))
            if self._transformer.arity != 1:
                raise ValueError('Invalid arity for `transformer`. Expected 1, '
                                 'got %d.' % (self._transformer.arity))

        params = self.get_params()
        params['_metric'] = self._metric

        params['tournament_size'] = self.tournament_size
        params['init_depth'] = self.init_depth
        params['init_method'] = self.init_method
        params['const_range'] = self.const_range
        params['parsimony_coefficient'] = self.parsimony_coefficient
        params['p_point_replace'] = self.p_point_replace
        params['max_samples'] = self.max_samples
        params['feature_names'] = self.feature_names

        if hasattr(self, '_transformer'):
            params['_transformer'] = self._transformer
        else:
            params['_transformer'] = None
        params['function_set'] = self._function_set
        params['arities'] = self._arities
        params['method_probs'] = self._method_probs
        const_flag = True
        if self.const_range == None:
            const_flag = False
        selected_space = select_space(cal_spacebound(
            self.function_set, self.n_features_, var_bound, const_flag=const_flag), low_bound, high_bound)
        params['selected_space'] = selected_space
        self.qualified_list = [
            self.qualified_list[-2], self.qualified_list[-1]]
        params['qualified_list'] = self.qualified_list
        params['eq_write'] = None
        n_samples, n_features = X.shape


        # print(selected_space)
        # print(len(selected_space))

        if not self.warm_start or not hasattr(self, '_programs'):
            # Free allocated memory, if any
            self._programs = []
            self.run_details_ = {'generation': [],
                                 'average_length': [],
                                 'average_fitness': [],
                                 'best_length': [],
                                 'best_fitness': [],
                                 'best_oob_fitness': [],
                                 'generation_time': []}

        prior_generations = len(self._programs)

        # print(self._programs)
        n_more_generations = self.generations - prior_generations

        print(n_more_generations)

        if n_more_generations < 0:
            raise ValueError('generations=%d must be larger or equal to '
                             'len(_programs)=%d when warm_start==True'
                             % (self.generations, len(self._programs)))
        elif n_more_generations == 0:
            fitness = [program.raw_fitness_ for program in self._programs[-1]]
            warn('Warm-start fitting without increasing n_estimators does not '
                 'fit new programs.')

        if self.warm_start:
            # Generate and discard seeds that would have been produced on the
            # initial fit call.
            for i in range(len(self._programs)):
                _ = random_state.randint(MAX_INT, size=self.population_size)

        # print(self._programs[1])

        if self.verbose:
            # Print header fields
            self._verbose_reporter()
        # 编写代码：1.父子代合并   2.非域排序   3.拥挤度排序
        best_program = None
        best_program_fitness_ = None

        # one = 1
        # population = None
        # for gen in range(one):
        #     top1Flag = False
        #     start_time = time()

        #     if gen == 0:
        #         parents = None
        #     else:
        #         parents = self._programs[gen - 1]
        #         print("xxx!")
        #         print(parents.__str__())
        #         parents.sort(key=lambda x: x.raw_fitness_)
        #         np.random.shuffle(parents)
        #         top1Flag = True
        #     n_jobs, n_programs, starts = _partition_estimators(
        #         self.population_size, self.n_jobs)
        seeds = random_state.randint(MAX_INT, size=self.population_size)

        #     population = Parallel(n_jobs=n_jobs,
        #                           verbose=int(self.verbose > 1))(
        #         delayed(_parallel_evolve)(n_programs[i],
        #                                   parents,
        #                                   X,
        #                                   y,
        #                                   self.sample_weight,
        #                                   seeds[starts[i]:starts[i + 1]],
        #                                   params)
        #         for i in range(n_jobs))

        #     # Reduce, maintaining order across different n_jobs
        #     population = list(itertools.chain.from_iterable(population))
        #     print("ttttttttt")

        #     print(population[110].__str__())
        #     gen += 1

        return X, y, params, self.population_size,seeds,self.qualified_list,self.function_set,n_features

        # for gen in range(prior_generations, self.generations):
        #     top1Flag = False
        #     start_time = time()

        #     if gen == 0:
        #         parents = None
        #     else:
        #         parents = self._programs[gen - 1]
        #         parents.sort(key=lambda x: x.raw_fitness_)
        #         np.random.shuffle(parents)
        #         top1Flag = True
        #     n_jobs, n_programs, starts = _partition_estimators(
        #         self.population_size, self.n_jobs)
        #     seeds = random_state.randint(MAX_INT, size=self.population_size)

        #     population = Parallel(n_jobs=n_jobs,
        #                           verbose=int(self.verbose > 1))(
        #         delayed(_parallel_evolve)(n_programs[i],
        #                                   parents,
        #                                   X,
        #                                   y,
        #                                   self.sample_weight,
        #                                   seeds[starts[i]:starts[i + 1]],
        #                                   params)
        #         for i in range(n_jobs))

        #     # Reduce, maintaining order across different n_jobs
        #     population = list(itertools.chain.from_iterable(population))
        #     print("ttttttttt")

        #     print(population[110].__str__())

        #     #父子代合并
        #     if parents is not None:
        #         population.extend(parents)
        #     #多目标优化中保存了父代和子代，所以不需要单独向种群中添加父代最优个体了
        #     # if top1Flag:
        #     #     population.append(best_program_fitness_)
        #     #     population.append(best_program)
        #     #快速非支配排序+多余front[k]的拥挤度排序--->筛选出新父代
        #     temp_index = self.fast_non_dominated_sort(population)
        #     '''
        #     for subPop in temp_index:
        #         prin = [population[i].raw_fitness_ for i in subPop]
        #         print(prin)
        #         prin = [population[i].length_ for i in subPop]
        #         print(prin)
        #     '''

        #     temp_popSize = 0
        #     population_index = []
        #     reminder_subPopulation = []
        #     for subPop in temp_index:
        #         pre_temp_popSize = temp_popSize
        #         temp_popSize += len(subPop)
        #         if temp_popSize >self.population_size:
        #             reminder = self.population_size-pre_temp_popSize
        #             # print("temp_popSize: ",temp_popSize,"reminder: ",reminder)
        #             reminder_subPopulation.extend(self.select_by_crowding_distance(population,subPop,reminder))
        #             # print("reminder_subPopulation: ",reminder_subPopulation)
        #             break
        #         else:
        #             population_index.extend(subPop)
        #     # print("len(population_index):",len(population_index),"population_index: ",population_index)
        #     # population_index = sum(temp_index , [])[:self.population_size]
        #     population = [population[i] for i in population_index]
        #     if reminder_subPopulation !=[]:
        #         population.extend(reminder_subPopulation)
        #     # print("实际种群数量=", len(population))
        #     #if gen % 100 ==0:
        #     #    print("实际种群数量=", len(population),population[0],population[200],population[400],population[-1])
        #     fitness = [program.raw_fitness_ for program in population]
        #     length = [program.length_ for program in population]
        #     # print(fitness,length,sep="\n")

        #     parsimony_coefficient = None
        #     if self.parsimony_coefficient == 'auto':
        #         parsimony_coefficient = (np.cov(length, fitness)[1, 0] /
        #                                  np.var(length))
        #     for program in population:
        #         program.fitness_ = program.fitness(parsimony_coefficient)
        #     fitness_ = [program.fitness_ for program in population]
        #     self._programs.append(population)

        #     # Remove old programs that didn't make it into the new population.
        #     if not self.low_memory:
        #         for old_gen in np.arange(gen, 0, -1):
        #             indices = []
        #             for program in self._programs[old_gen]:
        #                 if program is not None and program.parents is not None:
        #                     for idx in program.parents:
        #                         if 'idx' in idx:
        #                             indices.append(program.parents[idx])
        #             indices = set(indices)
        #             for idx in range(self.population_size):
        #                 if idx not in indices:
        #                     self._programs[old_gen - 1][idx] = None
        #     elif gen > 0:
        #         # Remove old generations
        #         self._programs[gen - 1] = None

        #     # # Record run details
        #     # if self._metric.greater_is_better:
        #     #     best_program = population[np.argmax(fitness)]#按惩罚项的fitness排序
        #     #     best_program_fitness_ = population[np.argmax(fitness_)]
        #     # else:
        #     #     best_program = population[np.argmin(fitness)]
        #     #     best_program_fitness_ = population[np.argmin(fitness_)]

        #     # self.run_details_['generation'].append(gen)
        #     # self.run_details_['average_length'].append(np.mean(length))
        #     # self.run_details_['average_fitness'].append(np.mean(fitness_))
        #     # self.run_details_['best_length'].append(best_program.length_)
        #     # self.run_details_['best_fitness'].append(best_program.fitness_)
        #     # oob_fitness = np.nan
        #     # if self.max_samples < 1.0:
        #     #     oob_fitness = best_program.oob_fitness_
        #     # self.run_details_['best_oob_fitness'].append(oob_fitness)
        #     # generation_time = time() - start_time
        #     # self.run_details_['generation_time'].append(generation_time)

        #     # if self.verbose:
        #     #     self._verbose_reporter(self.run_details_)

        #     # Check for early stopping
        #     if self._metric.greater_is_better:
        #         best_fitness = fitness[np.argmax(fitness_)]
        #         if best_fitness >= self.stopping_criteria:
        #             break
        #     else:
        #         best_fitness = fitness[np.argmin(fitness_)]
        #         if best_fitness <= self.stopping_criteria:
        #             break

        # if isinstance(self, TransformerMixin):
        #     # Find the best individuals in the final generation
        #     fitness = np.array(fitness_)
        #     if self._metric.greater_is_better:
        #         hall_of_fame = fitness.argsort()[::-1][:self.hall_of_fame]
        #     else:
        #         hall_of_fame = fitness.argsort()[:self.hall_of_fame]
        #     evaluation = np.array([gp.execute(X) for gp in
        #                            [self._programs[-1][i] for
        #                             i in hall_of_fame]])
        #     if self.metric == 'spearman':
        #         evaluation = np.apply_along_axis(rankdata, 1, evaluation)

        #     with np.errstate(divide='ignore', invalid='ignore'):
        #         correlations = np.abs(np.corrcoef(evaluation))
        #     np.fill_diagonal(correlations, 0.)
        #     components = list(range(self.hall_of_fame))
        #     indices = list(range(self.hall_of_fame))
        #     # Iteratively remove least fit individual of most correlated pair
        #     while len(components) > self.n_components:
        #         most_correlated = np.unravel_index(np.argmax(correlations),
        #                                            correlations.shape)
        #         # The correlation matrix is sorted by fitness, so identifying
        #         # the least fit of the pair is simply getting the higher index
        #         worst = max(most_correlated)
        #         components.pop(worst)
        #         indices.remove(worst)
        #         correlations = correlations[:, indices][indices, :]
        #         indices = list(range(len(components)))
        #     self._best_programs = [self._programs[-1][i] for i in
        #                            hall_of_fame[components]]

        # else:
        #     # Find the best individual in the final generation
        #     if self._metric.greater_is_better:
        #         self._program = self._programs[-1][np.argmax(fitness)]
        #     else:
        #         self._program = self._programs[-1][np.argmin(fitness)]
        # if  self._program.raw_fitness_ <self.global_fitness:
        #     self.sympy_global_best = sympify(self._program)
        #     self.global_fitness = self._program.raw_fitness_
        #     self.best_is_gp = True

        # return self
