import random
from abc import abstractmethod

import numpy as np

from keplar.operator.composite_operator import CompositeOp
from keplar.operator.reinserter import OperonReinserter, KeplarReinserter
import pyoperon as Operon


# class SR_Alg(CompositeOp):
#     def __init__(self,Comop_list):
#         super().__init__()
#         self.Comop_list = Comop_list

#     def do(self):
#         for Comop in self.Comop_list:
#             Comop.do()

class Alg:
    def __init__(self, max_generation, up_op_list, down_op_list, eva_op_list, error_tolerance, population):
        self.max_generation = max_generation
        self.up_op_list = up_op_list
        self.down_op_list = down_op_list
        self.eva_op_list = eva_op_list
        self.error_tolerance = error_tolerance
        self.population = population
        self.age = 0

    @abstractmethod
    def run(self):
        raise NotImplementedError


class uDSR_Alg(Alg):
    def __init__(self, max_generation, up_op_list, down_op_list, eva_op_list, error_tolerance, population):
        super().__init__(max_generation, up_op_list, down_op_list, eva_op_list, error_tolerance, population)


class BingoAlg(Alg):

    def __init__(self, max_generation, up_op_list, down_op_list, eva_op_list, error_tolerance, population):
        super().__init__(max_generation, up_op_list, down_op_list,
                         eva_op_list, error_tolerance, population)

    def get_best_fitness(self):
        return self.population.target_fit_list[self.get_best()]

    def get_best(self):
        best_fitness = self.population.target_fit_list[0]
        for j in range(len(self.population.target_fit_list)):
            if self.population.target_fit_list[j] < best_fitness:
                best_fitness = self.population.target_fit_list[j]
        return j

    def get_best_individual(self):
        return self.population.target_pop_list[self.get_best()]

    def run(self):
        generation_pop_size = self.population.get_pop_size()
        self.eva_op_list.do(self.population)
        now_error = self.get_best_fitness()
        while self.age < self.max_generation and now_error >= self.error_tolerance or str(now_error) == "nan":
            self.population = self.down_op_list.do(self.population)
            while generation_pop_size > self.population.get_pop_size():
                self.up_op_list.do(self.population)
            self.eva_op_list.do(self.population)
            now_error = self.get_best_fitness()
            best_ind = str(self.get_best_individual())
            self.age += 1
            print("第" + f"{self.age}代种群，" +
                  f"最佳个体适应度为{now_error}" + f"最佳个体为{best_ind}")
        best_ind = str(self.get_best_individual())
        print("迭代结束，共迭代" + f"{self.age}代" +
              f"最佳个体适应度为{now_error}" + f"最佳个体为{best_ind}")


class OperonBingoAlg(Alg):

    def __init__(self, max_generation, up_op_list, down_op_list, eva_op_list, error_tolerance, population, selector,
                 np_x, np_y):
        super().__init__(max_generation, up_op_list, down_op_list, eva_op_list, error_tolerance, population)
        self.selector = selector
        self.np_y = np_y.reshape([-1, 1])
        self.np_x = np_x
        self.ds = Operon.Dataset(np.hstack([np_x, self.np_y]))

    def get_tar_best(self):
        best_fitness = self.population.target_fit_list[0]
        for j in range(len(self.population.target_fit_list)):
            if self.population.target_fit_list[j] < best_fitness:
                best_fitness = self.population.target_fit_list[j]
        return j

    def get_best(self):
        best_fitness = self.population.pop_list[0].get_fitness()
        for j in range(len(self.population.pop_list)):
            if self.population.pop_list[j].get_fitness() < best_fitness:
                best_fitness = self.population.pop_list[j].get_fitness()
        return j

    def get_best_fitness(self):
        if self.population.pop_type != "self":
            best_num = self.get_tar_best()
            return self.population.target_fit_list[best_num]
        else:
            best_num = self.get_best()
            return self.population.pop_list[best_num].get_fitness()

    def get_best_individual(self):
        if self.population.pop_type != "self":
            print("牛牛牛")
            best_num = self.get_tar_best()
            str_op_tree = Operon.InfixFormatter.Format(self.population.target_pop_list[best_num], self.ds, 5)
            return str_op_tree
        else:
            print(self.population.pop_type)
            best_num = self.get_best()
            return self.population.pop_list[best_num].format()

    def run(self):
        for i in self.eva_op_list:
            i.do(self.population)
        now_error = self.get_best_fitness()
        while self.age < self.max_generation and now_error >= self.error_tolerance or str(now_error) == "nan":
            pool_list = self.selector.do(self.population)
            for i in self.up_op_list:
                i.do(pool_list)
            for i in self.eva_op_list:
                i.do(pool_list)
            reinserter = KeplarReinserter(pool_list, "self")
            reinserter.do(self.population)
            now_error = self.get_best_fitness()
            best_ind = str(self.get_best_individual())
            self.age += 1
            print("第" + f"{self.age}代种群，" +
                  f"最佳个体适应度为{now_error}" + f"最佳个体为{best_ind}")
        best_ind = str(self.get_best_individual())

        print("迭代结束，共迭代" + f"{self.age}代" +
              f"最佳个体适应度为{now_error}" + f"最佳个体为{best_ind}")
