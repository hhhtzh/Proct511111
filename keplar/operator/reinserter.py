import random

from keplar.operator.operator import Operator
import pyoperon as Operon


class Reinserter(Operator):
    def __init__(self):
        super().__init__()

    def do(self, population):
        raise NotImplementedError


class OperonReinserter(Reinserter):
    def __init__(self, pool_list, method, comparision_size, to_type):
        super().__init__()
        self.to_type = to_type
        self.pool_list = pool_list
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
        if population.pop_type == "Operon":
            ind_list = []
            if len(population.target_pop_list) != len(population.target_fit_list):
                raise ValueError("个体与适应度数量不符")
            for i in range(len(population.target_pop_list)):
                ind = Operon.Individual()
                ind.Genotype = population.target_pop_list[i]
                ind.Setfitness([population.target_fit_list[i]], 0)
                ind_list.append(ind)
            rein(rng, ind_list, self.pool_list)
            new_target_pop_list = []
            new_target_fitness_list = []
            for i in ind_list:
                new_target_pop_list.append(i.Genotype)
                new_target_fitness_list.append(i.GetFitness(0)[0])
            population.target_pop_list = new_target_pop_list
            population.target_fit_list = new_target_pop_list
            if self.to_type != "Operon":
                pass

        else:
            pass
