import random

from bingo.symbolic_regression import AGraph
from keplar.population.individual import Individual
from keplar.operator.operator import Operator
import numpy as np
from bingo.symbolic_regression.agraph.crossover import AGraphCrossover
from keplar.translator.translator import to_op, trans_op, to_gp, trans_taylor_program, taylor_trans_population
import pyoperon as Operon

from TaylorGP.src.taylorGP.functions import _Function
from TaylorGP.src.taylorGP._global import get_value


class Crossover(Operator):
    def __init__(self):
        super().__init__()

    def do(self, population):
        pass


class BingoCrossover(Crossover):
    def __init__(self, to_type):
        super().__init__()
        self.to_type = to_type

    def do(self, population):

        [parent_1_num, parent_2_num] = np.random.randint(low=0, high=population.get_pop_size() - 1, size=2)
        if population.pop_type != "Bingo":
            # print(population.pop_type)
            population.set_pop_size(len(population.pop_list))
            parent_1 = population.pop_list[parent_1_num]
            parent_2 = population.pop_list[parent_2_num]
            right = False
            while not right:
                [parent_1_num, parent_2_num] = np.random.randint(low=0, high=population.get_pop_size() - 1, size=2)
                parent_1 = population.pop_list[parent_1_num]
                parent_2 = population.pop_list[parent_2_num]
                # print(parent_1.format())
                self.bingo_parent_1 = AGraph(equation=str(parent_1.format()))
                self.bingo_parent_2 = AGraph(equation=str(parent_2.format()))
                if self.bingo_parent_2.command_array.shape == self.bingo_parent_1.command_array.shape and \
                        self.bingo_parent_1.command_array.shape[0] > 2 and self.bingo_parent_2.command_array.shape[
                    0] > 2:
                    right = True
                else:
                    right = False
        else:
            population.set_pop_size(len(population.target_pop_list))
            right = False
            num = 0
            while not right:
                if num > 20:
                    return
                [parent_1_num, parent_2_num] = np.random.randint(low=0, high=population.get_pop_size() - 1, size=2)
                self.bingo_parent_1 = population.target_pop_list[parent_1_num]
                requests_shape = self.bingo_parent_1.command_array.shape
                for i in population.target_pop_list:
                    # print(i.command_array)
                    # print(i.command_array.shape)
                    # print(requests_shape)
                    if i.command_array.shape == requests_shape:
                        self.bingo_parent_2 = i
                        break
                if self.bingo_parent_1.command_array.shape[0] > 2 and self.bingo_parent_2.command_array.shape[
                    0] > 2:
                    right = True
                else:
                    right = False
                    num += 1
        crossover = AGraphCrossover()
        self.bingo_parent_1._update()
        self.bingo_parent_2._update()
        bingo_child_1, bingo_child_2 = crossover(parent_1=self.bingo_parent_1, parent_2=self.bingo_parent_2)
        if self.to_type != "Bingo":
            child_1 = Individual(equation=str(bingo_child_1))
            child_2 = Individual(equation=str(bingo_child_2))
            population.append(child_1)
            population.append(child_2)
            new_pop_size = population.get_pop_size() + 2
            population.set_pop_size(new_pop_size)
        else:
            population.target_pop_list.append(bingo_child_1)
            population.target_pop_list.append(bingo_child_2)
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
        np_y = np_y.reshape([-1, 1])
        self.ds = Operon.Dataset(np.hstack([np_x, np_y]))
        self.np_x = np_x
        self.np_y = np_y

    def do(self, population):
        [parent_1_num, parent_2_num] = np.random.randint(low=0, high=population.get_pop_size() - 1, size=2)
        if population.pop_type != "Operon":
            pass
            parent_1 = population.pop_list[parent_1_num]
            parent_2 = population.pop_list[parent_2_num]
            op_parent1 = to_op(parent_1, np_x=self.np_x, np_y=self.np_y)
            op_parent2 = to_op(parent_2, np_x=self.np_x, np_y=self.np_y)
        else:
            op_parent1 = population.target_pop_list[parent_1_num]
            op_parent2 = population.target_pop_list[parent_2_num]
        crossover = Operon.SubtreeCrossover(self.internal_probability, self.depth_limit, self.length_limit)
        rng = Operon.RomuTrio(random.randint(1, 1000000))
        new_tree = crossover(rng, op_parent1, op_parent2)
        self.old_inds = []
        old_str1 = Operon.InfixFormatter.Format(op_parent1, self.ds, 5)
        old_str2 = Operon.InfixFormatter.Format(op_parent2, self.ds, 5)
        self.old_inds.append(old_str1)
        self.old_inds.append(old_str2)
        self.new_ind = Operon.InfixFormatter.Format(new_tree, self.ds, 5)
        if self.to_type != "Operon":
            population.self_pop_enable = True
            population.pop_type = "self"
            ind = trans_op(new_tree)
            population.pop_list.append(ind)
            new_pop_size = population.get_pop_size() + 1
            population.set_pop_size(new_pop_size)
        else:
            population.pop_type = "Operon"
            population.target_pop_list.append(new_tree)
            new_pop_size = len(population.target_pop_list)
            population.set_pop_size(new_pop_size)
            population.self_pop_enable = False


class TaylorGPCrossover(Crossover):
    def __init__(self, random_state, pop_parent, pop_honor, pop_now_index):
        super().__init__()
        # self.best_idx = best_idx
        self.random_state = random_state
        # self.qualified_list =qualified_list
        # self.pop_idx =pop_idx
        # self.function_set= function_set
        # self.n_features=n_features
        self.pop_parent = pop_parent
        self.pop_honor = pop_honor
        self.pop_now_index = pop_now_index
        # print("fffff")
        # print(self.pop_best_index)

    def get_value(self, random_state, pop_parent, pop_honor, pop_now_index):
        self.random_state = random_state
        # self.qualified_list =qualified_list
        # self.function_set= function_set
        # self.n_features=n_features
        self.pop_parent = pop_parent
        self.pop_honor = pop_honor
        self.pop_now_index = pop_now_index

    def do(self, population=None):

        # parent = trans_taylor_program(population.target_pop_list[self.pop_best_index])
        # donor = trans_taylor_program(population.target_pop_list[self.donor_best_index])

        # parent = trans_taylor_program(self.pop_parent)
        # honor = trans_taylor_program(self.pop_honor)
        # print(self.pop_parent.)

        program = None
        qualified_flag = False
        op_index = 0
        while qualified_flag == False:
            op_index = self.random_state.randint(5)
            if get_value('TUIHUA_FLAG'):
                break
            elif self.pop_parent.qualified_list == [1, -1] and (op_index == 2 or op_index == 3):
                continue
            elif self.pop_parent.qualified_list == [2, -1] and op_index == 4 and self.pop_parent.n_features > 1:
                continue
            elif (self.pop_parent.qualified_list == [-1, 1] or self.pop_parent.qualified_list == [-1, 2]) and (
                    op_index == 1 or op_index == 3):
                continue
            elif (self.pop_parent.qualified_list == [1, 1] or self.pop_parent.qualified_list == [-1, 2]) and (
                    op_index == 1 or op_index == 2 or op_index == 3):
                continue
            qualified_flag = True

        # print(op_index)
        # print(self.pop_now_index)

        if op_index < 4:
            # program = self.pop_parent.function_set[op_index:op_index + 1]
            program = self.pop_parent.function_set[op_index:op_index + 1] + self.pop_parent.program[:] + self.pop_honor[
                                                                                                         :]
            # population = taylor_trans_population(program,population,self.pop_now_index)=program
            # population.target_pop_list[self.pop_now_index].program=program
            # return self.pop_parent.program,None,None
            return program, None, None

            # return population
            # return  program,None,None
        else:
            x_index = self.random_state.randint(self.pop_parent.n_features)
            if x_index not in self.pop_parent.program:
                for i in range(len(self.pop_parent.program)):
                    if isinstance(self.pop_parent.program[i], int):
                        x_index = self.pop_parent.program[i]
                        break

            for node in range(len(self.pop_parent.program)):
                if isinstance(self.pop_parent.program[node], _Function) == False and self.pop_parent.program[
                    node] == x_index:
                    terminal = self.pop_honor
                    program = self.changeTo(self.pop_parent.program, node, terminal)
                    # population.target_pop_list[self.pop_now_index].program=program

            # return program,None,None
            # population = taylor_trans_population(program,population,self.pop_now_index)
            return program, None, None
            # return population

    def changeTo(self, program, node, terminal):
        return program[:node] + terminal + program[node + 1:]


class GplearnCrossover(Crossover):
    def __init__(self, to_type):
        super().__init__()
        self.to_type = to_type

    def do(self, population):
        if population.pop_type != "Gplearn":
            gp_pop = to_gp(population.pop_list)
