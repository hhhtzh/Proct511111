import random

from keplar.operator.operator import Operator
import pyoperon as Operon

from keplar.population.population import Population
from keplar.translator.translator import to_op


class Reinserter(Operator):
    def __init__(self):
        super().__init__()

    def do(self, population):
        raise NotImplementedError


class KeplarReinserter(Reinserter):
    def __init__(self, pool, to_type):
        super().__init__()
        self.to_type = to_type
        self.pool = pool

    def do(self, population):
        if population.pop_type == "self" and self.pool.pop_type == "self":
            # print("正确的")
            # print(population.pop_list)
            while len(self.pool.pop_list) != 0:
                # print(len(population.pop_list))
                new_ind = self.pool.pop_list.pop()
                # print(new_ind)
                # print(new_ind.format())
                population.pop_list.append(new_ind)
                # print(len(population.pop_list))
                # print(new_ind.format(), new_ind.get_fitness())
                worest_index = 0
                for i in range(len(population.pop_list)):
                    # print(new_ind)
                    # print(population.pop_list[i])
                    # print(population.pop_list[i].get_fitness())
                    # print(population.pop_list[worest_index].get_fitness())
                    if (population.pop_list[i].get_fitness()) and (
                            population.pop_list[worest_index].get_fitness()) and (
                            population.pop_list[i].get_fitness() > population.pop_list[worest_index].get_fitness()):
                        worest_index = i
                population.pop_list.pop(worest_index)
            # print(population.pop_list)
            if self.to_type != "Operon":
                population.pop_type = "self"


class OperonReinserter(Reinserter):
    def __init__(self, pool, method, comparision_size, to_type, np_x, np_y):
        super().__init__()
        self.np_y = np_y
        self.np_x = np_x
        self.to_type = to_type
        self.pool = pool
        self.comparision_size = comparision_size
        self.method = method

    def do(self, population):
        rng = Operon.RomuTrio(random.randint(1, 1000000))
        if self.comparision_size > population.pop_size:
            raise ValueError("比较数量大于种群数量")
        if not isinstance(self.comparision_size, int):
            raise ValueError("比较数量必须为int类型")
        if self.method == "ReplaceWorst":
            rein = Operon.ReplaceWorstReinserter(objective_index=0)
        elif self.method == "KeepBest":
            rein = Operon.KeepBestReinserter(objective_index=0)
        else:
            raise ValueError("reinserter方法选择错误")
        if population.pop_type == "Operon" and self.pool.pop_type == "Operon":
            ind_list = []
            if len(population.target_pop_list) != len(population.target_fit_list):
                raise ValueError("个体与适应度数量不符")
            for i in range(len(population.target_pop_list)):
                ind = Operon.Individual()
                ind.Genotype = population.target_pop_list[i]
                ind.Setfitness([population.target_fit_list[i]], 0)
                ind_list.append(ind)
            rein(rng, ind_list, self.pool)
            new_target_pop_list = []
            new_target_fitness_list = []
            for i in ind_list:
                new_target_pop_list.append(i.Genotype)
                new_target_fitness_list.append(i.GetFitness(0)[0])
            population.target_pop_list = new_target_pop_list
            population.target_fit_list = new_target_pop_list
            population.pop_type = "Operon"
            if self.to_type != "Operon":
                pass

        elif population.pop_type == "self" and self.pool.pop_type == "Operon":
            pop_ind_list = []
            pool_ind_list = []
            for i in range(len(population.pop_list)):
                self_ind = population.pop_list[i]
                op_tree = to_op(self_ind, self.np_x, self.np_y)
                ind = Operon.Individual()
                ind.Genotype = op_tree
                ind.SetFitness(population.pop_list[i].get_fitness(), 0)
                pop_ind_list.append(ind)
            for i in range(len(self.pool.target_pop_list)):
                op_tree = self.pool.target_pop_list[i]
                ind = Operon.Individual()
                ind.Genotype = op_tree
                ind.SetFitness(self.pool.target_fit_list[i], 0)
                pool_ind_list.append(ind)
            rein(rng, pop_ind_list, pool_ind_list)
            new_target_pop_list = []
            new_target_fitness_list = []
            for i in pop_ind_list:
                new_target_pop_list.append(i.Genotype)
                new_target_fitness_list.append(i.GetFitness(0)[0])
            population.target_pop_list = new_target_pop_list
            population.target_fit_list = new_target_pop_list
            population.pop_type = "Operon"
        elif population.pop_type == "self" and self.pool.pop_type == "self":
            pop_ind_list = []
            pool_ind_list = []
            for i in range(len(population.pop_list)):
                self_ind = population.pop_list[i]
                op_tree = to_op(self_ind, self.np_x, self.np_y)
                ind = Operon.Individual()
                ind.Genotype = op_tree
                ind.SetFitness(population.pop_list[i].get_fitness(), 0)
                pop_ind_list.append((ind, 1))
            for i in range(len(self.pool.pop_list)):
                self_ind = self.pool.pop_list[i]
                op_tree = to_op(self_ind, self.np_x, self.np_y)
                ind = Operon.Individual()
                ind.Genotype = op_tree
                ind.SetFitness(self.pool.pop_list[i].get_fitness(), 0)
                pool_ind_list.append((ind, 1))
            rein(rng, pop_ind_list, pool_ind_list)
            new_target_pop_list = []
            new_target_fitness_list = []
            for i in pop_ind_list:
                new_target_pop_list.append(i.Genotype)
                new_target_fitness_list.append(i.GetFitness(0)[0])
            population.target_pop_list = new_target_pop_list
            population.target_fit_list = new_target_pop_list
            population.pop_type = "Operon"


        else:
            print("true:" + str(population.pop_type) + str(self.pool.pop_type))


class TaylorGPReinserter(Reinserter):
    def __init__(self):
        super().__init__()

    def do(self, population=None):
        return super().do(population)
