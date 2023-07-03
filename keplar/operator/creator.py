import random
import time

import numpy as np

from TaylorGP.TaylorGP import CalTaylorFeatures
from TaylorGP.src.taylorGP.calTaylor import Metrics
from bingo.symbolic_regression.agraph.string_parsing import postfix_to_command_array_and_constants
from gplearn.genetic import SymbolicRegressor

from bingo.symbolic_regression import ComponentGenerator, AGraphGenerator, AGraph
from keplar.population.individual import Individual
from keplar.operator.operator import Operator

from keplar.population.population import Population
from keplar.translator.translator import trans_gp, trans_op, bingo_infixstr_to_func
import pyoperon as Operon
from TaylorGP.src.taylorGP.functions import _Function, _sympol_map


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
    def __init__(self, pop_size, x, y):
        super().__init__()
        self.pop_size = pop_size
        self.x = x
        self.y = y

    def do(self, population=None):
        pop = Population(self.pop_size)
        for i in range(self.pop_size):
            reg = SymbolicRegressor(generations=1, population_size=1)
            reg.fit(self.x, self.y)
            equ = trans_gp(str(reg))
            ind = Individual(str(equ))
            pop.append(ind)
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
    def __init__(self, program,to_type):
        super().__init__()
        self.program = program
        self.to_type = to_type


    def do(self, population=None):
        population_size = len(self.program)
        # print(population_size)
        population =Population(population_size)
        if self.to_type == "Taylor":    
            for i in range(population_size):
                eq = []

                for j, node in enumerate(self.program[i].program):
                    if isinstance(node, _Function):
                        eq.append(node.name)
                    else:
                        eq.append(node)
                # print(self.program[i].__str__())
                # print(eq)
                population.append(eq)
        else:
            pass

        return population
