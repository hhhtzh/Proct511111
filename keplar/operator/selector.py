from abc import abstractmethod

from bingo.selection.age_fitness import AgeFitness
from bingo.selection.deterministic_crowding import DeterministicCrowding
from bingo.selection.tournament import Tournament
from bingo.symbolic_regression import AGraph
from keplar.population.individual import Individual
from keplar.operator.operator import Operator
from keplar.population.population import Population


class Selector(Operator):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def do(self, population):
        raise NotImplementedError


class BingoSelector(Operator):
    def __init__(self, eliminate_p, method):
        super().__init__()
        self.method = method
        if 0 < eliminate_p < 1:
            self.eliminate_p = eliminate_p
        else:
            raise ValueError("淘汰比例数值错误")

    def do(self, population):
        target_pop_size = int(population.get_pop_size() * (1-self.eliminate_p))
        if self.method == "tournament":
            selector = Tournament(int(population.get_pop_size()/10))
        elif self.method == "age_fitness":
            selector = AgeFitness(10)
        elif self.method == "deterministic_crowding":
            selector = DeterministicCrowding()
        else:
            raise ValueError("选择方法错误")
        bingo_pop = []
        for i in population.pop_list:
            equation = i.equation
            bingo_ind = AGraph(equation=str(equation))
            bingo_ind.fitness=i.get_fitness()
            bingo_ind._update()
            bingo_pop.append(bingo_ind)
        new_bingo_pop = selector(population=bingo_pop, target_population_size=target_pop_size)
        new_pop = Population(len(new_bingo_pop))
        new_pop_list = []
        for i in new_bingo_pop:
            ind = Individual(equation=str(i))
            new_pop_list.append(ind)
        new_pop.initial(new_pop_list)
        return new_pop
