from bingo.symbolic_regression import AGraph
from keplar.population.individual import Individual
from keplar.operator.operator import Operator
import numpy as np
from bingo.symbolic_regression.agraph.crossover import AGraphCrossover


class Crossover(Operator):
    def __init__(self):
        super().__init__()

    def do(self, population):
        pass


class BingoCrossover(Crossover):
    def __init__(self):
        super().__init__()

    def do(self, population):
        population.set_pop_size(len(population.pop_list))
        [parent_1_num, parent_2_num] = np.random.randint(low=0, high=population.get_pop_size() - 1, size=2)
        parent_1 = population.pop_list[parent_1_num]
        parent_2 = population.pop_list[parent_2_num]
        right = False
        while not right:
            [parent_1_num, parent_2_num] = np.random.randint(low=0, high=population.get_pop_size() - 1, size=2)
            parent_1 = population.pop_list[parent_1_num]
            parent_2 = population.pop_list[parent_2_num]
            self.bingo_parent_1 = AGraph(equation=str(parent_1.equation))
            self.bingo_parent_2 = AGraph(equation=str(parent_2.equation))
            if self.bingo_parent_2.command_array.shape == self.bingo_parent_1.command_array.shape and \
                    self.bingo_parent_1.command_array.shape[0] > 2 and self.bingo_parent_2.command_array.shape[0] > 2:
                right = True
            else:
                right = False
        crossover = AGraphCrossover()
        self.bingo_parent_1._update()
        self.bingo_parent_2._update()
        bingo_child_1, bingo_child_2 = crossover(parent_1=self.bingo_parent_1, parent_2=self.bingo_parent_2)
        child_1 = Individual(equation=str(bingo_child_1))
        child_2 = Individual(equation=str(bingo_child_2))
        population.append(child_1)
        population.append(child_2)
        new_pop_size = population.get_pop_size() + 2
        population.set_pop_size(new_pop_size)


class BingoCross(Crossover):
    def __init__(self):
        super().__init__()

    def do(self, population):
        population.set_pop_size(len(population.pop_list))
        [parent_1_num, parent_2_num] = np.random.randint(low=0, high=population.get_pop_size() - 1, size=2)
        parent_1 = population.pop_list[parent_1_num]
        parent_2 = population.pop_list[parent_2_num]
