import numpy as np

from bingo.evaluation.evaluation import Evaluation
from bingo.local_optimizers.local_opt_fitness import LocalOptFitnessFunction
from bingo.local_optimizers.scipy_optimizer import ScipyOptimizer
from bingo.symbolic_regression import ExplicitTrainingData, ExplicitRegression, ImplicitRegression, \
    ImplicitTrainingData, AGraph
from keplar.operator.operator import Operator


class Evaluator(Operator):
    def __init__(self):
        super().__init__()


class BingoEvaluator(Operator):
    def __init__(self, x, fit, optimizer_method, y=None, dx_dt=None):
        super().__init__()
        self.x = x
        self.y = y
        self.dx_dt = dx_dt
        self.fit = fit
        self.optimizer_method = optimizer_method

    def do(self, population):

        if self.fit == "exp":
            training_data = ExplicitTrainingData(self.x, self.y)
            fitness = ExplicitRegression(training_data=training_data)
        elif self.fit == "imp":
            training_data = ImplicitTrainingData(x=self.x, dx_dt=self.dx_dt)
            fitness = ImplicitRegression(training_data=training_data)
        else:
            raise ValueError("evaluator方法类型未识别")
        if self.optimizer_method not in ["lm", "TNC", "BFGS", "L-BFGS-B", "CG", "SLSQP"]:
            raise ValueError("优化方法名称未识别")
        optimizer = ScipyOptimizer(fitness, method=self.optimizer_method)
        local_opt_fitness = LocalOptFitnessFunction(fitness, optimizer)
        evaluator = Evaluation(local_opt_fitness)
        bingo_pop = []
        for i in population.pop_list:
            equation = i.equation
            bingo_ind = AGraph(equation=str(equation))
            bingo_ind._update()
            # print("这是equation"+equation)
            # print("这是直接转化后的array"+str(bingo_ind.command_array))
            bingo_pop.append(bingo_ind)
        evaluator(population=bingo_pop)
        for i in range(len(bingo_pop)):
            population.pop_list[i].fitness = bingo_pop[i].fitness
            population.pop_list[i].evaluated = True
