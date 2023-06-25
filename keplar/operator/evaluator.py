import random

import numpy as np

from bingo.evaluation.evaluation import Evaluation
from bingo.local_optimizers.local_opt_fitness import LocalOptFitnessFunction
from bingo.local_optimizers.scipy_optimizer import ScipyOptimizer
from bingo.symbolic_regression import ExplicitTrainingData, ExplicitRegression, ImplicitRegression, \
    ImplicitTrainingData, AGraph
from keplar.operator.operator import Operator
import pyoperon as Operon

from keplar.population.individual import Individual
from keplar.translator.translator import trans_op


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
        if population.pop_type != "Bingo":
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
        else:
            bingo_pop = population.target_pop_list
            population.set_pop_size(len(bingo_pop))
            # for i in bingo_pop:
            #     print(str(i))
            evaluator(population=bingo_pop)
            for i in range(len(bingo_pop)):
                population.target_fit_list.append(bingo_pop[i].fitness)


class OperonEvaluator(Operator):
    def __init__(self, error_metric, np_x, np_y, training_p, if_linear_scaling, to_type):
        super().__init__()
        self.to_type = to_type
        self.if_linear_scaling = if_linear_scaling
        self.training_p = training_p
        self.error_metric = error_metric
        np_y = np_y.reshape([-1, 1])
        self.ds = Operon.Dataset(np.hstack([np_x, np_y]))

    def do(self, population):
        if not isinstance(self.if_linear_scaling, bool):
            raise ValueError("if_linear_scaling必须为bool类型")
        interpreter = Operon.Interpreter()
        if self.error_metric == "R2":
            error_metric = Operon.R2()
        elif self.error_metric == "MSE":
            error_metric = Operon.MES()
        elif self.error_metric == "NMSE":
            error_metric = Operon.NMES()
        elif self.error_metric == "RMSE":
            error_metric = Operon.RMES()
        elif self.error_metric == "MAE":
            error_metric = Operon.MAE()
        elif self.error_metric == "C2":
            error_metric = Operon.C2()
        else:
            ValueError("误差矩阵类型错误")
        target = self.ds.Variables[-1]
        inputs = Operon.VariableCollection(v for v in self.ds.Variables if v.Name != target.Name)
        rng = Operon.RomuTrio(random.randint(1, 1000000))
        training_range = Operon.Range(0, int(self.ds.Rows * self.training_p))
        test_range = Operon.Range(int(self.ds.Rows * self.training_p), self.ds.Rows)
        problem = Operon.Problem(self.ds, inputs, target.Name, training_range, test_range)
        evaluator = Operon.Evaluator(problem, interpreter, error_metric, self.if_linear_scaling)
        if population.pop_type == "Operon":
            tree_list = population.target_pop_list
            ind_list = []
            fit_list = []
            for i in tree_list:
                ind = Operon.Individual()
                ind.Genotype = i
                ind_list.append(ind)
            for i in ind_list:
                ea = evaluator(rng, i)
                fit_list.append(ea[0])
            if self.to_type == "Operon":
                population.target_fit_list = fit_list
                population.pop_type = "Operon"
                population.self_pop_enable = False
            else:
                operon_ind_list = population.target_pop_list
                var_list = self.ds.Variables
                kep_pop_list = []
                for i in range(len(operon_ind_list)):
                    func, const_array = trans_op(operon_ind_list[i], var_list)
                    kep_ind = Individual(func)
                    kep_ind.const_array = const_array
                    kep_ind.set_fitness(fit_list[i])
                    kep_pop_list.append(kep_ind)
                population.pop_list = kep_pop_list
                population.pop_type="self"
                # for i in population.pop_list:
                #     print(i.format(),i.get_fitness())
        else:
            pass

class TaylorGPEvaluator(Operator):
    def __init__(self):
        super().__init__()
    
    def do(self, population=None):
        return super().do(population)
