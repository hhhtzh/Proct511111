from gplearn.genetic import SymbolicRegressor

from bingo.symbolic_regression import ComponentGenerator, AGraphGenerator
from keplar.population.individual import Individual
from keplar.operator.operator import Operator

from keplar.population.population import Population
from keplar.translator.translator import trans_gp
# import pyoperon as Operon


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
            equ = trans_gp(str(reg))
            ind = Individual(str(equ))
            pop.append(ind)
        return pop


# class OperonCreator(Creator):
#     def __init__(self, minL=1, maxL=50, maxD=10):
#         super().__init__()
#         self.maxD = maxD
#         self.minL = minL
#         self.maxL = maxL

#     def do(self, population=None):
#         pset = Operon.PrimitiveSet()
#         pset.SetConfig(Operon.PrimitiveSet.TypeCoherent)
