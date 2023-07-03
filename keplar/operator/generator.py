from abc import abstractmethod
import numpy as np

from keplar.operator.crossover import Crossover
from keplar.operator.evaluator import Evaluator
from keplar.operator.mutation import Mutation
from keplar.operator.operator import Operator
from keplar.operator.selector import Selector


class Generator(Operator):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def do(self, population):
        raise NotImplementedError

class GpGenerator(Generator):
    def __init__(self):
        super().__init__()

    def do(self,population):
        pass


class OperonGenerator(Generator):
    def __init__(self):
        super().__init__()

    def do(self, population):
        pass


class BingoGenerator(Generator):
    def __init__(self, max_generation, up_op_list, down_op_list, evaluator, error_tolerance,population):
        super().__init__()
        self.max_generation = max_generation
        self.up_op_list = up_op_list
        self.down_op_list = down_op_list
        self.evaluator = evaluator
        self.error_tolerance = error_tolerance
        self.population=population
        self.age = 0

    # def __init__(self, initial_population, generation_pop_size, x, max_generation, error_tolerance, fit,
    #              optimizer_method,
    #              evaluator_type, eliminate_p, selector_type, selector_method, operators, y=None, dx_dt=None,
    #              command_probability=0.2,
    #              node_probability=0.2, parameter_probability=0.2,
    #              prune_probability=0.2, fork_probability=0.2):
    #     super().__init__()
    #     self.age = 0
    #     self.population = initial_population
    #     self.x = x
    #     self.y = y
    #     self.dx_dt = dx_dt
    #     self.generation_pop_size = generation_pop_size
    #     self.max_generation = max_generation
    #     self.error_tolerance = error_tolerance
    #     self.fit = fit
    #     self.optimizer_method = optimizer_method
    #     self.evaluator_typr = evaluator_type
    #     self.eliminate_p = eliminate_p
    #     self.selector_type = selector_type
    #     self.selector_method = selector_method
    #     self.operators = operators
    #     self.command_probability = command_probability
    #     self.node_probability = node_probability
    #     self.parameter_probability = parameter_probability
    #     self.prune_probability = prune_probability
    #     self.fork_probability = fork_probability

    def get_best_fitness(self):
        return self.get_best().get_fitness()

    def get_best(self):
        best_fitness = self.population.pop_list[0].get_fitness()
        for indv in self.population.pop_list:
            if indv.get_fitness() > best_fitness:
                best_fitness = indv.get_fitness()
        return indv

    def get_best_individual(self):

        return self.get_best().equation

    def do(self, population):
        self.population = population
        generation_pop_size = self.population.get_pop_size()
        self.evaluator.do(self.population)
        now_error = self.get_best_fitness()
        while self.age < self.max_generation and now_error >= self.error_tolerance or str(now_error) == "nan":
            for op in self.down_op_list:
                self.population = op.do(self.population)
            while generation_pop_size > self.population.get_pop_size():
                for op in self.up_op_list:
                    op.do(self.population)
            self.evaluator.do(self.population)
            now_error = self.get_best_fitness()
            best_ind = str(self.get_best_individual())
            self.age += 1
            print("第" + f"{self.age}代种群，" + f"最佳个体适应度为{now_error}" + f"最佳个体为{best_ind}")
        best_ind = str(self.get_best_individual())
        print("迭代结束，共迭代" + f"{self.age}代" + f"最佳个体适应度为{now_error}" + f"最佳个体为{best_ind}")

        # for ind in population:
        #     if not ind.evaluated:
        #         raise ValueError("迭代之前需要先进行至少一次评估操作")
        # now_error = self.error_tolerance + 100
        # evaluator = Evaluator(x=self.x, y=self.y, dx_dt=self.dx_dt, population=self.population)
        # if evaluator_type == "bingo":
        #     evaluator.bingo_do(fit, optimizer_method)
        # else:
        #     raise ValueError("Evaluator类型未识别")
        # while self.age <= max_generation and now_error > error_tolerance:
        #     selector = Selector(eliminate_p, self.population)
        #     if selector_type == "bingo":
        #         selector.bingo_do(selector_method)
        #     else:
        #         raise ValueError("Selector类型未识别")
        #     while self.generation_pop_size <= self.population.get_pop_size():
        #         crossover = Crossover()
        #         if evaluator_type == "bingo":
        #             crossover.bingo_do(self.population)
        #         else:
        #             raise ValueError("Crossover类型未识别")
        #         mutation = Mutation()
        #         if evaluator_type == "bingo":
        #             mutation.bingo_do(self.population, self.x, operators, command_probability, node_probability
        #                               , parameter_probability, prune_probability, fork_probability)
        #         else:
        #             raise ValueError("Mutation类型未识别")
        #     if evaluator_type == "bingo":
        #         evaluator.bingo_do(fit, optimizer_method)
        #     else:
        #         raise ValueError("Evaluator类型未识别")
        #     now_error = self.get_best_fitness()
        #     self.age += 1
        #     print("第" + f"{self.age}代种群，" + f"最佳个体适应度为{now_error}")
        # best_ind = str(self.get_best_individual())
        # print("迭代结束，共迭代" + f"{self.age}代" + f"最佳个体适应度为{now_error}" + f"最佳个体为{best_ind}")
