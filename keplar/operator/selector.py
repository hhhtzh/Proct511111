import array
import random
from abc import abstractmethod

from bingo.selection.age_fitness import AgeFitness
from bingo.selection.deterministic_crowding import DeterministicCrowding
from bingo.selection.tournament import Tournament
from bingo.symbolic_regression import AGraph
from keplar.population.individual import Individual
from keplar.operator.operator import Operator
from keplar.population.population import Population
import pyoperon as Operon


class Selector(Operator):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def do(self, population):
        raise NotImplementedError


class BingoSelector(Operator):
    def __init__(self, select_p, method):
        super().__init__()
        self.method = method
        if 0 < select_p < 1:
            self.select_p = select_p
        else:
            raise ValueError("淘汰比例数值错误")

    def do(self, population):
        target_pop_size = int(population.get_pop_size() * (self.select_p))
        if self.method == "tournament":
            selector = Tournament(int(population.get_pop_size() / 10))
        elif self.method == "age_fitness":
            selector = AgeFitness(10)
        elif self.method == "deterministic_crowding":
            selector = DeterministicCrowding()
        else:
            raise ValueError("选择方法错误")
        bingo_pop = []
        equation_list=[]
        if population.pop_type!="Bingo":
            for i in population.pop_list:
                equ=
        for i in population.pop_list:
            equation = i.equation
            bingo_ind = AGraph(equation=str(equation))
            bingo_ind.fitness = i.get_fitness()
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


class OperonSelector(Selector):
    def __init__(self, method, compare_size, to_type):
        super().__init__()
        self.compare_size = compare_size
        self.to_type = to_type
        self.method = method

    def do(self, population):
        if self.method == "Tournament":
            sel = Operon.TournamentSelector(self.compare_size)
        elif self.method == "RankTournament":
            sel = Operon.RankTournamentSelector(self.compare_size)
        elif self.method == "Proportional":
            sel = Operon.ProportionalSelector(self.compare_size)
        elif self.method == "Random":
            sel = Operon.RandomSelector(self.compare_size)
        else:
            raise ValueError("selector方法输入错误")
        rng = Operon.RomuTrio(random.randint(1, 1000000))
        if population.pop_type == "Operon":
            ind_list = {}
            if len(population.target_pop_list) != len(population.target_fit_list):
                raise ValueError("个体与适应度数量不符")
            for i in range(len(population.target_pop_list)):
                ind = Operon.Individual()
                ind.Genotype = population.target_pop_list[i]
                ind.SetFitness(population.target_fit_list[i], 0)
                ind_list.
            print(type(ind_list[0]))
            sel.Prepare(ind_list)
            best_num = sel(rng)
            pool = Population(1)
        else:
            pass
        if self.to_type == "Operon":
            pool.target_pop_list.append(population.target_pop_list[best_num])
            pool.pop_type = "Operon"
            return pool
        else:
            pass
