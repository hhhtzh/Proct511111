from abc import abstractmethod

from bingo.symbolic_regression import AGraph
from keplar.operator.operator import Operator


class Cleaner(Operator):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def do(self, population):
        raise NotImplementedError


class BingoCleaner(Cleaner):
    def __init__(self):
        super().__init__()

    def do(self, population):
        for ind_num in range(len(population.pop_list)):
            equ = population.pop_list[ind_num].equation
            bingo_ind = AGraph(equation=str(equ))
            bingo_ind._update()
            test_ind = bingo_ind.copy()
            print("d")
            if test_ind.mutable_command_array.shape != bingo_ind.command_array.shape:
                print("x")
                del population[ind_num]
            return population
