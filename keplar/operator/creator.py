import random
import time

import numpy as np
from sklearn.base import RegressorMixin
from joblib import Parallel, delayed

from TaylorGP.TaylorGP import CalTaylorFeatures
from TaylorGP.src.taylorGP.calTaylor import Metrics

from bingo.symbolic_regression.agraph.string_parsing import postfix_to_command_array_and_constants
from gplearn.fitness import _Fitness, _fitness_map
from gplearn.genetic import SymbolicRegressor, BaseSymbolic, MAX_INT
from  gplearn.genetic import _parallel_evolve as gp_pe
from gplearn.utils import _partition_estimators, check_random_state
from keplar.population.function import _function_map
from bingo.symbolic_regression import ComponentGenerator, AGraphGenerator, AGraph
from keplar.population.individual import Individual
from keplar.operator.operator import Operator

from keplar.population.population import Population
from keplar.translator.translator import trans_gp, trans_op, bingo_infixstr_to_func, to_bingo,trans_taylor_program
import pyoperon as Operon
from TaylorGP.src.taylorGP.functions import _Function, _sympol_map
from joblib import Parallel, delayed #自动创建进程池执行并行化操作
import itertools
from TaylorGP.src.taylorGP.genetic import alarm_handler, MAX_INT, _parallel_evolve, BaseEstimator, BaseSymbolic
from TaylorGP.src.taylorGP._program import _Program,print_program
import math





class Creator(Operator):
    def __init__(self):
        super().__init__()

    def do(self, population):
        pass


class BingoCreator(Creator):
    def __init__(self, pop_size, operators, x, stack_size, to_type):
        super().__init__()
        self.to_type = to_type
        self.pop_size = pop_size
        self.operators = operators
        self.population = []
        self.x = x
        self.stack_size = stack_size

    def do(self, population=None):
        component_generator = ComponentGenerator(self.x.shape[1])
        for op in self.operators:
            component_generator.add_operator(op)
        agraph_generator = AGraphGenerator(self.stack_size, component_generator)
        self.population = Population(pop_size=self.pop_size)
        pop = [agraph_generator() for _ in range(self.pop_size)]
        pop_list = []
        if self.to_type == "Bingo":
            self.population.target_pop_list = pop
            self.population.pop_type = "Bingo"
        else:
            self.population.pop_type = "self"
            for bingo_agraph in pop:
                bingo_str = str(bingo_agraph)
                func_list, const_array = bingo_infixstr_to_func(bingo_str)
                keplar_ind = Individual(func_list, const_array)
                pop_list.append(keplar_ind)
            self.population.pop_list = pop_list
            self.population.set_pop_size(len(pop_list))
        return self.population


class GpCreator(Creator):
    def __init__(self, pop_size, x, y, to_type, operators=('add', 'sub', 'mul', 'div'), init_depth=(2, 6),
                 init_method="half and half",
                 p_point_replace=0.05, sample_weight=None, metric='mean absolute error', const_range=(-1., 1.),
                 parsimony_coefficient=0.001, n_jobs=1, random_state=None):
        super().__init__()
        self.random_state = random_state
        self.n_jobs = n_jobs
        self._arities = None
        self._function_set = None
        self.parsimony_coefficient = parsimony_coefficient
        self.p_point_replace = p_point_replace
        self._metric = None
        self.metric = metric
        self.const_range = const_range
        self.init_method = init_method
        self.init_depth = init_depth
        self.operators = operators
        self.sample_weight = sample_weight
        self.to_type = to_type
        self.pop_size = pop_size
        self.x = x
        self.y = y

    def do(self, population=None):
        if isinstance(self.metric, _Fitness):
            self._metric = self.metric
        elif self.metric in ('mean absolute error', 'mse', 'rmse',
                             'pearson', 'spearman'):
            self._metric = _fitness_map[self.metric]
        else:
            raise ValueError("metric必须为gplearn的_Fitness实例")

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

        self._function_set = []
        for function in self.operators:
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
        gp_sy = SymbolicRegressor(function_set=self.operators, parsimony_coefficient=self.parsimony_coefficient,
                                  const_range=self.const_range, init_depth=self.init_depth, init_method=self.init_method
                                  , p_point_replace=self.p_point_replace)
        params = gp_sy.get_params()
        params['_metric'] = self._metric
        params['function_set'] = self._function_set
        params['method_probs'] = None
        params['_transformer'] = None
        params['selected_space'] = None
        params['qualified_list']=None
        self._arities = {}
        for function in self._function_set:
            arity = function.arity
            self._arities[arity] = self._arities.get(arity, [])
            self._arities[arity].append(function)
        params['arities'] = self._arities
        pop = Population(self.pop_size)
        n_jobs, n_programs, starts = _partition_estimators(
            self.pop_size, self.n_jobs)
        random_state = check_random_state(self.random_state)
        seeds = random_state.randint(MAX_INT, size=self.pop_size)
        parents = None
        population = Parallel(n_jobs=n_jobs,
                              verbose=True)(
            delayed(gp_pe)(n_programs[i],
                                      parents,
                                      self.x,
                                      self.y,
                                      self.sample_weight,
                                      seeds[starts[i]:starts[i + 1]],
                                      params)
            for i in range(n_jobs))
        for i in population:
            for j in i:
                pop.target_pop_list.append(j)
        pop.pop_type = "gplearn"

        if self.to_type != "gplearn":
            for gp_program in pop.target_pop_list:
                ind = trans_gp(gp_program)
                pop.pop_list.append(ind)
                pop.pop_type = "self"
                pop.pop_size = len(pop.pop_list)
        return pop


class OperonCreator(Creator):
    def __init__(self, tree_type, np_x, np_y, pop_size, to_type, minL=1, maxL=50, maxD=10, decimalPrecision=5):
        super().__init__()
        if tree_type == "balanced" or tree_type == "probabilistic":
            self.tree_type = tree_type
        else:
            raise ValueError("创建树的类型错误")
        self.maxD = maxD
        self.minL = minL
        self.maxL = maxL
        self.to_type = to_type
        self.pop_size = pop_size
        self.decimalPrecision = decimalPrecision
        np_y = np_y.reshape([-1, 1])
        self.ds = Operon.Dataset(np.hstack([np_x, np_y]))
        self.target = self.ds.Variables[-1]
        self.inputs = Operon.VariableCollection(v for v in self.ds.Variables if v.Name != self.target.Name)

    def do(self, population=None):
        pset = Operon.PrimitiveSet()
        pset.SetConfig(Operon.PrimitiveSet.TypeCoherent)
        if self.tree_type == "balanced":
            tree_creator = Operon.BalancedTreeCreator(pset, self.inputs, bias=0.0)
        elif self.tree_type == "probabilistic":
            tree_creator = Operon.ProbabilisticTreeCreator(pset, self.inputs, bias=0.0)
        else:
            raise ValueError("Operon创建树的类型名称错误")
        tree_initializer = Operon.UniformLengthTreeInitializer(tree_creator)
        tree_initializer.ParameterizeDistribution(self.minL, self.maxL)
        tree_initializer.MaxDepth = self.maxD
        rng = Operon.RomuTrio(random.randint(1, 1000000))
        coeff_initializer = Operon.NormalCoefficientInitializer()
        coeff_initializer.ParameterizeDistribution(0, 1)
        tree_list = []
        pop = Population(self.pop_size)
        pop.pop_type = "Operon"

        variable_list = self.ds.Variables
        for i in range(self.pop_size):
            tree = tree_initializer(rng)
            coeff_initializer(rng, tree)
            tree_list.append(tree)
        pop.check_flag(self.to_type)
        trans_flag = pop.translate_flag
        pop.target_pop_list = tree_list
        if trans_flag:
            for i in tree_list:
                func = trans_op(i, variable_list)
                ind_new = Individual(func=func)
                pop.append(ind_new)
                pop.self_pop_enable = True
        return pop


class uDSR_Creator(Creator):
    def __init__(self, T):
        super().__init__()
        self.T = T
        self.population = None

    def do(self, population=None):
        if self.T is None:
            print("T is None")
        else:
            length_T = len(self.T)
            # l = np.array([len(T[i]) for i in range(length_T)])
            self.population = Population(length_T)
            self.population.pop_type = "uDSR"
            self.population.target_pop_list = self.T
            return self.population


class TaylorGPCreator(Creator):
    def __init__(self, X,y,params,gen,population_size,program,to_type):
        super().__init__()
        self.X= X
        self.y=y
        self.params = params
        self.gen=gen
        self.to_type = to_type
        self.random_state =None
        self.population_size=population_size
        self.n_jobs =1
        self.verbose = 0
        self.sample_weight = None
        self.program =program




    def do(self, population=None):

        population = Population(self.population_size)

        random_state = check_random_state(self.random_state)

        n_jobs, n_programs, starts = _partition_estimators(
                self.population_size, self.n_jobs)
        seeds = random_state.randint(MAX_INT, size=self.population_size)
        n_samples, n_features = self.X.shape
        # Unpack parameters
        tournament_size = self.params['tournament_size']
        function_set = self.params['function_set']
        arities = self.params['arities']
        init_depth = self.params['init_depth']
        init_method = self.params['init_method']
        const_range = self.params['const_range']
        metric = self.params['_metric']
        transformer = self.params['_transformer']
        parsimony_coefficient = self.params['parsimony_coefficient']
        method_probs = self.params['method_probs']
        p_point_replace = self.params['p_point_replace']
        max_samples = self.params['max_samples']
        feature_names = self.params['feature_names']
        selected_space = self.params['selected_space']
        qualified_list = self.params['qualified_list'] #合格判别标准
        eq_write = self.params['eq_write'] #用于将所有生成的公式写入eq_write文件

        max_samples = int(max_samples * n_samples)

        for i in range(self.population_size):
            genome = None
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
                            program=self.program,
                            selected_space = selected_space,
                            qualified_list = qualified_list,
                            X =self.X,
                            eq_write =  eq_write)
            program.parents = genome

            # Draw samples, using sample weights, and then fit
            if self.sample_weight is None:
                curr_sample_weight = np.ones((n_samples,))
            else:
                curr_sample_weight = self.sample_weight.copy()
            oob_sample_weight = curr_sample_weight.copy()
            
            indices, not_indices = program.get_all_indices(n_samples,
                                                        max_samples,
                                                        random_state)

            curr_sample_weight[not_indices] = 0
            oob_sample_weight[indices] = 0

            program.raw_fitness_ = program.raw_fitness(self.X, self.y, curr_sample_weight)
            if math.isnan(program.raw_fitness_) or math.isinf(program.raw_fitness_) or program.length_ >500:
                i -= 1
                continue
            program.fitness_ = program.fitness(parsimony_coefficient)
            # if max_samples < n_samples:
            #     # Calculate OOB fitness
            #     program.oob_fitness_ = program.raw_fitness(self.X, self.y, oob_sample_weight)
            
            eq=[]
            if self.to_type == "Taylor":
                for i, node in enumerate(program.program):
                    if isinstance(node, _Function):
                        eq.append(node.name)
                    else:
                        eq.append(node)
                ind=Individual(eq)
                # ind.fitness = program.raw_fitness_
                population.target_append(ind)
                idx =population.pop_size-1
                population.target_fit_list[idx]=program.fitness_

            else:
                pass


        return population

