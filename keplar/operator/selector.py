import array
import random
import re
from abc import abstractmethod

from bingo.selection.age_fitness import AgeFitness
from bingo.selection.deterministic_crowding import DeterministicCrowding
from bingo.selection.tournament import Tournament
from bingo.symbolic_regression import AGraph
from keplar.population.individual import Individual
from keplar.operator.operator import Operator
from keplar.population.population import Population
from keplar.translator.translator import bingo_infixstr_to_func

import pyoperon as Operon

from keplar.translator.translator import bingo_infixstr_to_func
import numpy as np


class Selector(Operator):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def do(self, population):
        raise NotImplementedError


class BingoSelector(Operator):
    def __init__(self, select_p, method, to_type):
        super().__init__()
        self.to_type = to_type
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
        equation_list = []
        if population.pop_type != "Bingo":
            for i in population.pop_list:
                equ_list = i.format()
                # print(equ_list)
                equ_list = re.sub(r"pow", r"^", equ_list)
                bingo_ind = AGraph(equation=equ_list)
                bingo_ind.fitness = i.get_fitness()
                bingo_ind._update()
                bingo_pop.append(bingo_ind)
        else:
            bingo_pop = population.target_pop_list
        # print(population.pop_type)
        new_bingo_pop = selector(population=bingo_pop, target_population_size=target_pop_size)
        new_pop = Population(len(new_bingo_pop))
        if self.to_type != "Bingo":
            new_pop_list = []
            for i in new_bingo_pop:
                equ = str(i)
                func, const_array = bingo_infixstr_to_func(equ)
                ind = Individual(func)
                ind.const_array = const_array
                ind.set_fitness(i.fitness)
                new_pop_list.append(ind)
            new_pop.initial(new_pop_list)
            new_pop.pop_type = "self"
        else:
            new_pop.pop_type = "Bingo"

            new_pop.target_pop_list = new_bingo_pop
        return new_pop


class OperonSelector(Selector):
    def __init__(self, select_p, method, to_type):
        super().__init__()
        self.to_type = to_type
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
        equation_list = []
        if population.pop_type != "Bingo":
            for i in population.pop_list:
                equ_list = i.format()
                # print(equ_list)
                bingo_ind = AGraph(equation=equ_list)
                bingo_ind.fitness = i.get_fitness()
                bingo_ind._update()
                bingo_pop.append(bingo_ind)
        else:
            bingo_pop = population.target_pop_list
        # print(population.pop_type)
        new_bingo_pop = selector(population=bingo_pop, target_population_size=target_pop_size)
        new_pop = Population(len(new_bingo_pop))
        if self.to_type != "Bingo":
            new_pop_list = []
            for i in new_bingo_pop:
                equ = str(i)
                func, const_array = bingo_infixstr_to_func(equ)
                ind = Individual(func)
                ind.const_array = const_array
                ind.set_fitness(i.fitness)
                new_pop_list.append(ind)
            new_pop.initial(new_pop_list)
            new_pop.pop_type = "self"
        else:
            new_pop.pop_type = "Bingo"
            new_pop.target_pop_list = new_bingo_pop
        return new_pop


class TaylorGPSelector(Selector):
    def __init__(self, random_state, tournament_size, greater_is_better):
        super().__init__()
        self.random_state = random_state
        self.tournament_size = tournament_size
        self.greater_is_better = greater_is_better

    def get_value(self, random_state, tournament_size):
        self.random_state = random_state
        self.tournament_size = tournament_size
        # self.greater_is_better =greater_is_better

    def do(self, population=None):
        # print("8888888")
        # print(population.pop_size)
        contenders = self.random_state.randint(1, population.pop_size, self.tournament_size)
        # print(contenders)
        # for p in contenders:
        #     print(p)
        #     print(population.target_fit_list[p])
        # print("kkkk")
        # print(population.target_fit_list[999])
        # print("kkkk")
        # print(contenders)
        # print(population.pop_size)
        # print(population.target_pop_list[999].fitness_)
        # print(population.target_pop_list[1].fitness_)

        fitness = [population.target_pop_list[p].raw_fitness_ for p in contenders]
        if self.greater_is_better:
            parent_index = contenders[np.argmax(fitness)]
        else:
            parent_index = contenders[np.argmin(fitness)]
        # print("77777")
        return population.target_pop_list[parent_index], parent_index


class GpSelector(Selector):
    def __init__(self, random_state, tournament_size, greater_is_better=None):
        super().__init__()
        self.random_state = random_state
        self.tournament_size = tournament_size
        self.greater_is_better = greater_is_better

    def get_value(self, random_state, tournament_size):
        self.random_state = random_state
        self.tournament_size = tournament_size
        # self.greater_is_better =greater_is_better

    def do(self, population):
        # print("8888888")
        # print(population.pop_size)
        contenders = self.random_state.randint(0, population.pop_size, self.tournament_size)
        # print(contenders)
        # for p in contenders:
        #     print(p)
        #     print(population.target_fit_list[p])
        # print("kkkk")
        # print(population.target_fit_list[999])
        # print("kkkk")
        # print(contenders)
        # print(population.pop_size)
        # print(population.target_pop_list[999].fitness_)
        # print(population.target_pop_list[1].fitness_)

        fitness = [population.target_pop_list[p].raw_fitness_ for p in contenders]
        if self.greater_is_better:
            parent_index = contenders[np.argmax(fitness)]
        else:
            parent_index = contenders[np.argmin(fitness)]
        # print("77777")
        return population.target_pop_list[parent_index], parent_index
