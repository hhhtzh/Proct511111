import random

from bingo.symbolic_regression import AGraph
from keplar.population.individual import Individual
from keplar.operator.operator import Operator
import numpy as np
from bingo.symbolic_regression.agraph.crossover import AGraphCrossover
from keplar.translator.translator import to_op, trans_op
import pyoperon as Operon


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


class OperonCrossover(Crossover):
    def __init__(self, np_x, np_y, to_type, internal_probability=0.9, depth_limit=10, length_limit=50):
        self.to_type = to_type
        self.internal_probability = internal_probability
        self.depth_limit = depth_limit
        self.length_limit = length_limit
        self.np_x = np_x
        self.np_y = np_y.reshape([-1, 1])
        super().__init__()

    def do(self, population):
        [parent_1_num, parent_2_num] = np.random.randint(low=0, high=population.get_pop_size() - 1, size=2)
        if population.pop_type != "Operon":
            pass
            # parent_1 = population.pop_list[parent_1_num]
            # parent_2 = population.pop_list[parent_2_num]
            # op_parent1 = to_op(parent_1, np_x=self.np_x, np_y=self.np_y)
            # op_parent2 = to_op(parent_2, np_x=self.np_x, np_y=self.np_y)
        else:
            op_parent1 = population.target_pop_list[parent_1_num]
            op_parent2 = population.target_pop_list[parent_2_num]
        crossover = Operon.SubtreeCrossover(self.internal_probability, self.depth_limit, self.length_limit)
        rng = Operon.RomuTrio(random.randint(1, 1000000))
        new_tree = crossover(rng, op_parent1, op_parent2)
        if self.to_type != "Operon":
            population.self_pop_enable = True
            population.pop_type = "self"
            ind = trans_op(new_tree)
            population.append(ind)
            new_pop_size = population.get_pop_size() + 1
            population.set_pop_size(new_pop_size)
        else:
            population.target_pop_list.append(new_tree)
            new_pop_size = len(population.target_pop_list)
            population.set_pop_size(new_pop_size)
            population.self_pop_enable = False
