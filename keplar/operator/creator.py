from gplearn.genetic import SymbolicRegressor

from bingo.symbolic_regression import ComponentGenerator, AGraphGenerator
from keplar.population.individual import Individual
from keplar.operator.operator import Operator

from keplar.population.population import Population
from keplar.translator.translator import trans_gp, Dsr2pop
# import pyoperon as Operon
import numpy as np


class Creator(Operator):
    def __init__(self):
        super().__init__()

    def do(self, population):
        pass


class BingoCreator(Creator):
    def __init__(self, pop_size, operators, x, stack_size):
        super().__init__()
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
        for i in pop:
            ind = Individual(equation=str(i))
            pop_list.append(ind)
        self.population.initial(pop_list=pop_list)
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
            print(str(reg))

            equ = trans_gp(str(reg))
            ind = Individual(str(equ))
            pop.append(ind)
        return pop
    
class DsrCreator(Creator):
    def __init__(self, programs):
        super().__init__()
        self.programs = programs
    def do(self, population=None):
        r = np.array([p.r for p in self.programs])
        l = np.array([len(p.traversal) for p in self.programs])
        expr = np.array([p.sympy_expr for p in self.programs])
        # T = np.array([p.traversal for p in self.programs])
        # print(T[0])
        # print(l[0])
        # print(expr[0])
        pop_size = len(expr)
        population=Population(pop_size)
        # express = {}
        for i in range(pop_size):
            # print(expr[i])
            # express[i]= Dsr2pop(str(expr[i]))
            # print(express[i])
            ind = Individual(expr[i])
            population.append(ind)

        population.set_pop_size(pop_size)

        return population


# class OperonCreator(Creator):
#     def __init__(self, minL=1, maxL=50, maxD=10):
#         super().__init__()
#         self.maxD = maxD
#         self.minL = minL
#         self.maxL = maxL

#     def do(self, population=None):
#         pset = Operon.PrimitiveSet()
#         pset.SetConfig(Operon.PrimitiveSet.TypeCoherent)
