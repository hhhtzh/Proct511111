import random
import re

import numpy as np

from TaylorGP.src.taylorGP.fitness import _mean_square_error, _weighted_spearman, _log_loss, _mean_absolute_error, \
    _Fitness
from bingo.evaluation.evaluation import Evaluation
from bingo.local_optimizers.local_opt_fitness import LocalOptFitnessFunction
from bingo.local_optimizers.scipy_optimizer import ScipyOptimizer
from bingo.symbolic_regression import ExplicitTrainingData, ExplicitRegression, ImplicitRegression, \
    ImplicitTrainingData, AGraph
from bingo.symbolic_regression.agraph.string_parsing import eq_string_to_infix_tokens, infix_to_postfix
from gplearn._program import _Program
from gplearn.fitness import _root_mean_square_error
from keplar.operator.operator import Operator
import pyoperon as Operon
# import xgboost as xgb
from keplar.population.individual import Individual
from keplar.translator.translator import trans_op, to_bingo, trans_gp, bingo_to_gp, bgpostfix_to_gpprefix, equ_to_op, \
    bingo_infixstr_to_func, to_op


class Evaluator(Operator):
    def __init__(self):
        super().__init__()


class BingoEvaluator(Evaluator):
    def __init__(self, x, fit, optimizer_method, to_type, y=None, dx_dt=None, metric="rmse"):
        super().__init__()
        self.metric = metric
        self.to_type = to_type
        self.x = x
        self.y = y
        self.dx_dt = dx_dt
        self.fit = fit
        self.optimizer_method = optimizer_method

    def do(self, population):
        if self.fit == "exp":
            training_data = ExplicitTrainingData(self.x, self.y)
            fitness = ExplicitRegression(training_data=training_data, metric=self.metric)
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
                bingo_ind = to_bingo(i)
                bingo_pop.append(bingo_ind)
            evaluator(population=bingo_pop)
            for i in range(len(bingo_pop)):
                population.pop_list[i].fitness = bingo_pop[i].fitness
                population.pop_list[i].evaluated = True
            if self.to_type == "Bingo":
                population.target_pop_list = bingo_pop
                population.pop_type = "Bingo"
                population.target_fit_list = []
                # print("@@")
                # print(len(bingo_pop))
                for i in range(len(bingo_pop)):
                    population.target_fit_list.append(bingo_pop[i].fitness)

        else:
            bingo_pop = population.target_pop_list
            population.set_pop_size(len(bingo_pop))
            # for i in bingo_pop:
            #     print(str(i))
            evaluator(population=bingo_pop)
            population.target_fit_list=[]
            for i in range(len(bingo_pop)):
                population.target_fit_list.append(bingo_pop[i].fitness)
            if self.to_type != "Bingo":
                for i in range(len(bingo_pop)):
                    func, const_arr = bingo_infixstr_to_func(str(bingo_pop[i]))
                    ind = Individual(func=func, const_array=const_arr)
                    ind.set_fitness(bingo_pop[i].fitness)
                    population.pop_list.append(ind)
                population.pop_type = "self"

    def part_do(self, population):
        if self.fit == "exp":
            training_data = ExplicitTrainingData(self.x, self.y)
            fitness = ExplicitRegression(training_data=training_data, metric=self.metric)
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
                bingo_ind = to_bingo(i)
                bingo_pop.append(bingo_ind)
            evaluator(population=bingo_pop)
            for i in range(len(bingo_pop)):
                population.pop_list[i].fitness = bingo_pop[i].fitness
                population.pop_list[i].evaluated = True
            if self.to_type == "Bingo":
                population.target_pop_list = bingo_pop
                population.pop_type = "Bingo"
                for i in range(len(bingo_pop)):
                    population.target_fit_list[i] = bingo_pop[i].fitness

        else:
            bingo_pop = population.target_pop_list
            population.set_pop_size(len(bingo_pop))
            # for i in bingo_pop:
            #     print(str(i))
            evaluator(population=bingo_pop)
            for i in range(len(bingo_pop)):
                population.target_fit_list.append(bingo_pop[i].fitness)
            if self.to_type != "Bingo":
                for i in range(len(bingo_pop)):
                    func, const_arr = bingo_infixstr_to_func(str(bingo_pop[i]))
                    ind = Individual(func=func, const_array=const_arr)
                    ind.set_fitness(bingo_pop[i].fitness)
                    population.pop_list.append(ind)
                population.pop_type = "self"


class OperonEvaluator(Evaluator):
    def __init__(self, error_metric, np_x, np_y, training_p, if_linear_scaling, to_type):
        super().__init__()
        self.to_type = to_type
        self.if_linear_scaling = if_linear_scaling
        self.training_p = training_p
        self.error_metric = error_metric
        np_y = np_y.reshape([-1, 1])
        self.ds = Operon.Dataset(np.hstack([np_x, np_y]))
        self.np_x = np_x
        self.np_y = np_y

    def do(self, population):
        if not isinstance(self.if_linear_scaling, bool):
            raise ValueError("if_linear_scaling必须为bool类型")
        interpreter = Operon.Interpreter()
        if self.error_metric == "R2":
            error_metric = Operon.R2()
        elif self.error_metric == "MSE":
            error_metric = Operon.MSE()
        elif self.error_metric == "NMSE":
            error_metric = Operon.NMSE()
        elif self.error_metric == "RMSE":
            error_metric = Operon.RMSE()
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
            # print(len(population.target_pop_list))
            # print("***")
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
            # print(fit_list)
            if self.to_type == "Operon":
                population.target_fit_list = fit_list
                population.pop_type = "Operon"
                population.self_pop_enable = False
            else:
                # print("这块错了")
                operon_ind_list = population.target_pop_list
                var_list = self.ds.Variables
                kep_pop_list = []
                for i in range(len(operon_ind_list)):
                    func, const_array = trans_op(tree_list[i], var_list)
                    kep_ind = Individual(func)
                    kep_ind.const_array = const_array
                    # print(fit_list[i])
                    kep_ind.set_fitness(fit_list[i])
                    kep_pop_list.append(kep_ind)
                population.pop_list = kep_pop_list
                population.target_pop_list=[]
                population.target_fit_list=[]
                population.pop_type = "self"
                # for i in population.pop_list:
                #     print(i.format(),i.get_fitness())
        else:
            # print("jjj")
            tree_list = []
            ind_list = []
            fit_list = []
            for ind in population.pop_list:
                # print(ind.format())
                tree = to_op(ind, self.np_x, self.np_y)
                # str1 = Operon.InfixFormatter.Format(tree, self.ds, 5)
                tree_list.append(tree)
            for i in tree_list:
                ind = Operon.Individual()
                ind.Genotype = i
                ind_list.append(ind)
            for i in ind_list:
                ea = evaluator(rng, i)
                # print(ea)
                fit_list.append(ea[0])
            # for i in range(len(fit_list)):
            #     print(population.pop_list[i].format())
            #     print(population.pop_list[i].func)
            #     print(fit_list[i])
            if self.to_type == "Operon":
                population.target_fit_list = fit_list
                population.pop_type = "Operon"
                population.self_pop_enable = False
            else:
                operon_tree_list = tree_list
                var_list = self.ds.Variables
                kep_pop_list = []
                for i in range(len(operon_tree_list)):

                    # print(str1)
                    func, const_array = trans_op(operon_tree_list[i], var_list)
                    # print(func)
                    kep_ind = Individual(func)
                    kep_ind.const_array = const_array
                    kep_ind.set_fitness(fit_list[i])
                    kep_pop_list.append(kep_ind)
                population.pop_list = kep_pop_list
                population.pop_type = "self"


class OperonDiversityEvaluator(Evaluator):
    def __init__(self, np_x, np_y, training_p, if_linear_scaling, to_type):
        super().__init__()
        self.to_type = to_type
        self.if_linear_scaling = if_linear_scaling
        self.training_p = training_p
        np_y = np_y.reshape([-1, 1])
        self.ds = Operon.Dataset(np.hstack([np_x, np_y]))
        self.np_x = np_x
        self.np_y = np_y

    def do(self, population):
        if not isinstance(self.if_linear_scaling, bool):
            raise ValueError("if_linear_scaling必须为bool类型")
        target = self.ds.Variables[-1]
        inputs = Operon.VariableCollection(v for v in self.ds.Variables if v.Name != target.Name)
        rng = Operon.RomuTrio(random.randint(1, 1000000))
        training_range = Operon.Range(0, int(self.ds.Rows * self.training_p))
        test_range = Operon.Range(int(self.ds.Rows * self.training_p), self.ds.Rows)
        problem = Operon.Problem(self.ds, inputs, target.Name, training_range, test_range)
        evaluator = Operon.DiversityEvaluator(problem)
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
            # print(fit_list)
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
                    # print(fit_list[i])
                    kep_ind.set_fitness(fit_list[i])
                    kep_pop_list.append(kep_ind)
                population.pop_list = kep_pop_list
                population.pop_type = "self"
                # for i in population.pop_list:
                #     print(i.format(),i.get_fitness())
        else:
            tree_list = []
            ind_list = []
            fit_list = []
            for ind in population.pop_list:
                # print(ind.format())
                tree = to_op(ind, self.np_x, self.np_y)
                tree_list.append(tree)
            for i in tree_list:
                ind = Operon.Individual()
                ind.Genotype = i
                ind_list.append(ind)
            for i in ind_list:
                ea = evaluator(rng, i)
                # print(ea)
                fit_list.append(ea[0])
            # for i in range(len(fit_list)):
            #     print(population.pop_list[i].format())
            #     print(population.pop_list[i].func)
            #     print(fit_list[i])
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
                population.pop_type = "self"


class OperonSingleEvaluator(Evaluator):
    def __init__(self, error_metric, np_x, np_y, training_p, if_linear_scaling, op_equ):
        super().__init__()
        self.op_equ = op_equ
        self.if_linear_scaling = if_linear_scaling
        self.training_p = training_p
        self.error_metric = error_metric
        np_y = np_y.reshape([-1, 1])
        self.ds = Operon.Dataset(np.hstack([np_x, np_y]))
        self.np_x = np_x
        self.np_y = np_y

    def do(self, population=None):
        print("11")
        if not isinstance(self.if_linear_scaling, bool):
            raise ValueError("if_linear_scaling必须为bool类型")
        interpreter = Operon.Interpreter()
        if self.error_metric == "R2":
            error_metric = Operon.R2()
        elif self.error_metric == "MSE":
            error_metric = Operon.MSE()
        elif self.error_metric == "NMSE":
            error_metric = Operon.NMSE()
        elif self.error_metric == "RMSE":
            error_metric = Operon.RMSE()
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
        ind = Operon.Individual()
        self.op_equ = re.sub(r'X(\d{3})', r'X_\1', self.op_equ)
        self.op_equ_ = re.sub(r'X(\d{2})', r'X_\1', self.op_equ)
        self.op_equ_ = re.sub(r'X(\d{1})', r'X_\1', self.op_equ)
        equ = re.sub(r'x_', r'X_', self.op_equ)

        # 使用正则表达式查找科学计数法表示

        # 输入字符串包含科学计数法表示
        # 使用正则表达式查找科学计数法表示

        # 输入字符串包含科学计数法表示
        # 使用正则表达式查找科学计数法表示

        # 输入字符串包含科学计数法表示
        # 使用正则表达式查找科学计数法表示
        formula = equ.replace("**", "^")

        func_list, const_array = bingo_infixstr_to_func(formula)
        # print(func_list)
        ind1 = Individual(func=func_list, const_array=const_array)
        tree = to_op(ind1, self.np_x, self.np_y)
        ind.Genotype = tree
        ea = evaluator(rng, ind)
        fit = ea[0]
        return fit


class GpEvaluator(Evaluator):
    def __init__(self, eval_x, eval_y, to_type, feature_weight=None, metric="rmse"):
        self.to_type = to_type
        self.feature_weight = feature_weight
        super().__init__()
        self.eval_y = np.array(eval_y)
        self.eval_x = np.array(eval_x)
        self.metric = metric

    def do(self, population):
        if self.metric == "mse":
            fct = _mean_square_error
        elif self.metric == "rmse":
            fct = _root_mean_square_error
        elif self.metric == "log":
            fct = _log_loss
        elif self.metric == "mae":
            fct = _mean_absolute_error
        else:
            raise ValueError("gplearn评估模块计算误差方法设置错误")

        if population.pop_type == "gplearn":
            fit_list = []
            gp_fit = _Fitness(fct, False)
            lie_num = self.eval_x.shape[0]
            if self.feature_weight is None:
                self.feature_weight = []
                for i in range(lie_num):
                    self.feature_weight.append(1)
                self.feature_weight = np.array(self.feature_weight)
            for program in population.target_pop_list:
                pred_y = program.execute(self.eval_x).reshape(-1, 1)
                eva_y = self.eval_y.reshape(-1, 1)
                fw = self.feature_weight.reshape(-1, 1)
                # print(np.shape(eva_y))
                # print(np.shape(pred_y))
                # print(np.shape(fw))
                fitness = gp_fit(eva_y, pred_y.reshape(-1, 1), fw)
                fit_list.append(fitness)

            population.target_fit_list = fit_list
            if self.to_type != "gplearn":
                # print("ooooooooooooooooo")
                population.pop_type = "self"
                # print(len(population.target_pop_list))
                # print(len(population.target_fit_list))
                for i in range(len(population.target_pop_list)):
                    ind = trans_gp(population.target_pop_list[i])
                    ind.fitness = population.target_fit_list[i]
                    population.pop_list.append(ind)
        elif population.pop_type == "Bingo":
            # print("ooooooooooooooooooo")
            fit_list = []
            gp_fit = _Fitness(fct, False)
            lie_num = self.eval_x.shape[0]
            if self.feature_weight is None:
                self.feature_weight = []
                for i in range(lie_num):
                    self.feature_weight.append(1)
                self.feature_weight = np.array(self.feature_weight)
            gp_programs = []
            for program in population.target_pop_list:
                tk = bingo_to_gp(str(program))
                tk = eq_string_to_infix_tokens(str(tk))
                # print(tk)
                tk = infix_to_postfix(tk)
                # print(tk)
                tk = bgpostfix_to_gpprefix(tk)
                # print(tk)
                gp_prog = _Program(function_set=["add", "sub", "mul", "div", "sqrt", "neg", "power", "sin"],
                                   arities={"add": 2, "sub": 2, "mul": 2, "div": 2, "sqrt": 1, "neg": 1,
                                            "power": 2, "sin": 1},
                                   init_depth=[3, 3], init_method="half and half", n_features=4, const_range=[0, 1],
                                   metric="rmse",
                                   p_point_replace=0.4, parsimony_coefficient=0.01, random_state=1, program=tk)
                gp_programs.append(gp_prog)
            for program in gp_programs:
                eva_y = self.eval_y.reshape(-1, 1)
                fw = self.feature_weight.reshape(-1, 1)
                pred_y = program.execute(self.eval_x).reshape(-1, 1)
                fitness = gp_fit(eva_y, pred_y, fw)
                fit_list.append(fitness)
            population.target_pop_list = gp_programs
            population.target_fit_list = fit_list
            if self.to_type != "gplearn":
                # print("jjjjjjjjjjjjjjjjjjjjjjjj")
                for i, gp_program in enumerate(population.target_pop_list):
                    ind = trans_gp(gp_program)
                    ind.fitness = population.target_fit_list[i]
                    population.pop_list.append(ind)
                    population.pop_type = "self"
                    population.pop_size = len(population.pop_list)
        else:
            pass


class TaylorGPEvaluator(Evaluator):
    def __init__(self, method, eval_x, eval_y, to_type, feature_weight=None):
        self.to_type = to_type
        self.feature_weight = feature_weight
        super().__init__()
        self.eval_y = np.array(eval_y)
        self.eval_x = np.array(eval_x)
        self.method = method

    def do(self, population):
        if self.method == "mse":
            fct = _mean_square_error
        elif self.method == "rmse":
            fct = _weighted_spearman
        elif self.method == "log":
            fct = _log_loss
        elif self.method == "mae":
            fct = _mean_absolute_error
        else:
            raise ValueError("gplearn评估模块计算误差方法设置错误")

        if population.pop_type == "taylorgp":
            fit_list = []
            gp_fit = _Fitness(fct, False)
            lie_num = self.eval_x.shape[1]
            if self.feature_weight is None:
                self.feature_weight = []
                for i in range(lie_num):
                    self.feature_weight.append(1)
                self.feature_weight = np.array(self.feature_weight)
            z = 0
            for program in population.target_pop_list:
                pred_y = program.execute(self.eval_x)
                fitness = gp_fit(self.eval_y, pred_y, self.feature_weight)
                fit_list.append(fitness)
                z += 1
                print(z)
            if self.to_type == "taylorgp":
                population.target_fit_list = fit_list
            else:
                population.pop_type = "self"
                for i in range(len(population.target_pop_list)):
                    ind = trans_gp(population.target_pop_list[i])
                    population.pop_list.append(ind)
        else:
            pass

        return population


class MetricsBingoEvaluator(Evaluator):
    def __init__(self, data, func_fund_list, metric="rmse", optimizer_method="lm"):
        super().__init__()
        self.optimizer_method = optimizer_method
        self.metric = metric
        self.func_fund_list = func_fund_list
        self.data = data

    def do(self, population=None):
        x = self.data.get_np_x()
        y = self.data.get_np_y()
        np_xtrain = None
        training_data = ExplicitTrainingData(x, y)
        fitness = ExplicitRegression(training_data=training_data, metric=self.metric)
        if self.optimizer_method not in ["lm", "TNC", "BFGS", "L-BFGS-B", "CG", "SLSQP"]:
            raise ValueError("优化方法名称未识别")
        optimizer = ScipyOptimizer(fitness, method=self.optimizer_method)
        local_opt_fitness = LocalOptFitnessFunction(fitness, optimizer)
        evaluator = Evaluation(local_opt_fitness)
        for i in range(len(self.func_fund_list)):
            graph_list = []
            fit_arr = []
            for j in self.func_fund_list[i]:
                bingo_ind = AGraph(equation=str(j))
                bingo_ind._update()
                graph_list.append(bingo_ind)
            evaluator(graph_list)
            for j in graph_list:
                fit_arr.append(j.fitness)
            np_arr = np.array(fit_arr).reshape(1, -1)
            if i == 0:
                np_xtrain = np_arr
            else:
                np_xtrain = np.vstack((np_xtrain, np_arr))
        return np_xtrain


class SingleBingoEvaluator(Evaluator):
    def __init__(self, data, equation, metric="rmse", optimizer_method="lm"):
        super().__init__()
        self.equation = equation
        self.optimizer_method = optimizer_method
        self.metric = metric
        self.data = data

    def do(self, population=None):
        bingo_ind = AGraph(equation=self.equation)
        bingo_pop = [bingo_ind]
        x = self.data.get_np_x()
        y = self.data.get_np_y()
        training_data = ExplicitTrainingData(x, y)
        fitness = ExplicitRegression(training_data=training_data, metric=self.metric)
        if self.optimizer_method not in ["lm", "TNC", "BFGS", "L-BFGS-B", "CG", "SLSQP"]:
            raise ValueError("优化方法名称未识别")
        optimizer = ScipyOptimizer(fitness, method=self.optimizer_method)
        local_opt_fitness = LocalOptFitnessFunction(fitness, optimizer)
        evaluator = Evaluation(local_opt_fitness)
        evaluator(bingo_pop)
        bingo_ind = bingo_pop[0]
        bingo_ind._update()
        return bingo_ind.fitness


class XgBoostEvaluator(Evaluator):
    def __init__(self, eval_x, eval_y, to_type, feature_weight=None, metric="rmse"):
        self.to_type = to_type
        self.feature_weight = feature_weight
        super().__init__()
        self.eval_y = np.array(eval_y)
        self.eval_x = np.array(eval_x)
        self.metric = metric

    def do(self, population):
        if self.metric == "mse":
            fct = _mean_square_error
        elif self.metric == "rmse":
            fct = _root_mean_square_error
        elif self.metric == "log":
            fct = _log_loss
        elif self.metric == "mae":
            fct = _mean_absolute_error
        else:
            raise ValueError("gplearn评估模块计算误差方法设置错误")

        if population.pop_type == "gplearn":
            fit_list = []
            gp_fit = _Fitness(fct, False)
            lie_num = self.eval_x.shape[0]
            if self.feature_weight is None:
                self.feature_weight = []
                for i in range(lie_num):
                    self.feature_weight.append(1)
                self.feature_weight = np.array(self.feature_weight)
            for program in population.target_pop_list:
                pred_y = program.execute(self.eval_x).reshape(-1, 1)
                eva_y = self.eval_y.reshape(-1, 1)
                fw = self.feature_weight.reshape(-1, 1)
                # print(np.shape(eva_y))
                # print(np.shape(pred_y))
                # print(np.shape(fw))

                # # 将数学表达式编译为可执行的Python函数
                # func = individual.compile()
                #
                # # 计算表达式的预测值
                # y_pred = [func(*x) for x in X]
                #
                # # 使用XGBoost模型评估性能
                # model = xgb.XGBRegressor()
                # model.fit(X_train, y_train)
                # y_pred_xgb = model.predict(X_test)
                # mse = mean_squared_error(y_test, y_pred_xgb)
                #
                # fitness = gp_fit(eva_y, pred_y.reshape(-1, 1), fw)
                # fit_list.append(fitness)

            population.target_fit_list = fit_list
            if self.to_type != "gplearn":
                # print("ooooooooooooooooo")
                population.pop_type = "self"
                # print(len(population.target_pop_list))
                # print(len(population.target_fit_list))
                for i in range(len(population.target_pop_list)):
                    ind = trans_gp(population.target_pop_list[i])
                    ind.fitness = population.target_fit_list[i]
                    population.pop_list.append(ind)
        elif population.pop_type == "Bingo":
            # print("ooooooooooooooooooo")
            fit_list = []
            gp_fit = _Fitness(fct, False)
            lie_num = self.eval_x.shape[0]
            if self.feature_weight is None:
                self.feature_weight = []
                for i in range(lie_num):
                    self.feature_weight.append(1)
                self.feature_weight = np.array(self.feature_weight)
            gp_programs = []
            for program in population.target_pop_list:
                tk = bingo_to_gp(str(program))
                tk = eq_string_to_infix_tokens(str(tk))
                # print(tk)
                tk = infix_to_postfix(tk)
                # print(tk)
                tk = bgpostfix_to_gpprefix(tk)
                # print(tk)
                gp_prog = _Program(function_set=["add", "sub", "mul", "div", "sqrt", "neg", "power", "sin"],
                                   arities={"add": 2, "sub": 2, "mul": 2, "div": 2, "sqrt": 1, "neg": 1,
                                            "power": 2, "sin": 1},
                                   init_depth=[3, 3], init_method="half and half", n_features=4, const_range=[0, 1],
                                   metric="rmse",
                                   p_point_replace=0.4, parsimony_coefficient=0.01, random_state=1, program=tk)
                gp_programs.append(gp_prog)
            for program in gp_programs:
                eva_y = self.eval_y.reshape(-1, 1)
                fw = self.feature_weight.reshape(-1, 1)
                pred_y = program.execute(self.eval_x).reshape(-1, 1)
                fitness = gp_fit(eva_y, pred_y, fw)
                fit_list.append(fitness)
            population.target_pop_list = gp_programs
            population.target_fit_list = fit_list
            if self.to_type != "gplearn":
                # print("jjjjjjjjjjjjjjjjjjjjjjjj")
                for i, gp_program in enumerate(population.target_pop_list):
                    ind = trans_gp(gp_program)
                    ind.fitness = population.target_fit_list[i]
                    population.pop_list.append(ind)
                    population.pop_type = "self"
                    population.pop_size = len(population.pop_list)
        else:
            pass
