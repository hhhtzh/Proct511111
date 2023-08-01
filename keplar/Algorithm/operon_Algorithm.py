import random
import sys
import time

import numpy as np

from keplar.Algorithm.Alg import Alg
from keplar.data.data import Data
from keplar.operator.JudgeUCB import KeplarJudgeUCB
from keplar.operator.creator import OperonCreator
import pyoperon as Operon

from keplar.operator.crossover import OperonCrossover
from keplar.operator.evaluator import OperonEvaluator, SingleBingoEvaluator
from keplar.operator.mutation import OperonMutation
from keplar.operator.reinserter import OperonReinserter
from keplar.operator.selector import OperonSelector
from keplar.operator.sparseregression import KeplarSpareseRegression
from keplar.operator.statistic import BingoStatistic
from keplar.operator.taylor_judge import TaylorJudge
from keplar.posoperator.pruning import MOperonPruning

from keplar.preoperator.MTaylor.cluster_judge import ClusterJudge
from keplar.preoperator.ml.sklearndbscan import SklearnDBscan, SklearnDBscan1
from keplar.preoperator.ml.sklearnkmeans import SklearnKmeans
from keplar.translator.translator import trans_op1, trans_op2, trans_op0
from keplar.utils.utils import find_k_smallest_elements

gen = 0
max_ticks = 50

t0 = time.time()


class OperonAlg(Alg):
    def __init__(self, max_generation, up_op_list, down_op_list, eval_op_list, sel,
                 error_tolerance, population_size, threads_num, np_x, np_y, data, x_shape, minL=1, maxL=50, maxD=10,
                 decimalPrecision=5, population=None):
        super().__init__(max_generation, up_op_list, down_op_list, eval_op_list, error_tolerance, population)
        self.x_shape = x_shape
        self.data = data
        self.elapse_time = None
        self.best_fit = None
        self.model_fit = None
        self.model_string = None
        self.population_size = population_size
        self.threads_num = threads_num
        self.sel = sel
        self.minL = minL
        self.maxL = maxL
        self.maxD = maxD
        self.decimalPrecision = decimalPrecision
        self.np_x = np_x
        self.np_y = np_y
        np_y = np_y.reshape([-1, 1])
        self.ds = Operon.Dataset(np.hstack([np_x, np_y]))
        self.ind_list = []

    def get_n_top(self, n=3):
        result = find_k_smallest_elements(self.ind_list, n)
        for i in result:
            print(i.GetFitness(0))
            print(Operon.InfixFormatter.Format(i.Genotype, self.ds, 6))
        return result

    def run(self):
        t = time.time()
        selector = self.sel.do()
        for i in self.ds.Variables:
            if i.Index == self.x_shape:
                target = i
        # for i in self.ds.Variables:
        #     print(i.Name)
        #     print(i.Index)
        eva = self.eval_op_list[0]
        training_range = Operon.Range(0, int(self.ds.Rows * eva.training_p))
        test_range = Operon.Range(int(self.ds.Rows * eva.training_p), self.ds.Rows)
        interpreter = Operon.Interpreter()
        inputs = Operon.VariableCollection(v for v in self.ds.Variables if v.Name != target.Name)

        rng = Operon.RomuTrio(random.randint(1, 1000000))
        problem = Operon.Problem(self.ds, inputs, target.Name, training_range, test_range)
        if eva.error_metric == "R2":
            error_metric = Operon.R2()
        elif eva.error_metric == "MSE":
            error_metric = Operon.MSE()
        elif eva.error_metric == "NMSE":
            error_metric = Operon.NMSE()
        elif eva.error_metric == "RMSE":
            error_metric = Operon.RMSE()
        elif eva.error_metric == "MAE":
            error_metric = Operon.MAE()
        elif eva.error_metric == "C2":
            error_metric = Operon.C2()
        else:
            ValueError("误差矩阵类型错误")
        evaluator = Operon.Evaluator(problem, interpreter, error_metric,
                                     True)  # initialize evaluator, use linear scaling = True
        evaluator.Budget = 10000 * 10000  # computational budget
        evaluator.LocalOptimizationIterations = 0
        mut = self.up_op_list[0]
        cro = self.up_op_list[1]
        rein = self.down_op_list[0]
        pset = Operon.PrimitiveSet()
        pset.SetConfig(
            Operon.PrimitiveSet.TypeCoherent)
        if mut.tree_type == "balanced":
            tree_creator = Operon.BalancedTreeCreator(pset, inputs, bias=0.0)
        elif mut.tree_type == "probabilistic":
            tree_creator = Operon.ProbabilisticTreeCreator(pset, inputs, bias=0.0)
        else:
            raise ValueError("Operon创建树的类型名称错误")
        config = Operon.GeneticAlgorithmConfig(generations=self.max_generation, max_evaluations=1000000,
                                               local_iterations=0,
                                               population_size=self.population_size, pool_size=1000, p_crossover=1.0,
                                               p_mutation=0.25,
                                               epsilon=1e-5, seed=1, time_limit=86400)
        coeff_initializer = Operon.NormalCoefficientInitializer()
        coeff_initializer.ParameterizeDistribution(0, 1)
        mut_onepoint = Operon.NormalOnePointMutation()
        mut_changeVar = Operon.ChangeVariableMutation(inputs)
        mut_changeFunc = Operon.ChangeFunctionMutation(pset)
        mut_replace = Operon.ReplaceSubtreeMutation(tree_creator, coeff_initializer, self.maxD, self.maxL)
        mutation = Operon.MultiMutation()
        mutation.Add(mut_onepoint, mut.onepoint_p)
        mutation.Add(mut_changeVar, mut.changevar_p)
        mutation.Add(mut_changeFunc, mut.changefunc_p)
        mutation.Add(mut_replace, mut.replace_p)
        crossover = Operon.SubtreeCrossover(cro.internal_probability, cro.depth_limit, cro.length_limit)
        generator = Operon.BasicOffspringGenerator(evaluator, crossover, mutation, selector, selector)
        if rein.comparision_size > self.population_size:
            raise ValueError("比较数量大于种群数量")
        if not isinstance(rein.comparision_size, int):
            raise ValueError("比较数量必须为int类型")
        if rein.method == "ReplaceWorst":
            reinsert = Operon.ReplaceWorstReinserter(rein.comparision_size)
        elif rein.method == "KeepBest":
            reinsert = Operon.KeepBestReinserter(rein.comparision_size)
        else:
            raise ValueError("reinserter方法选择错误")
        tree_initializer = Operon.UniformLengthTreeInitializer(tree_creator)
        tree_initializer.ParameterizeDistribution(self.minL, self.maxL)
        tree_initializer.MaxDepth = self.maxD
        gp = Operon.GeneticProgrammingAlgorithm(problem, config, tree_initializer, coeff_initializer, generator,
                                                reinsert)
        interval = 1 if config.Generations < max_ticks else int(np.round(config.Generations / max_ticks, 0))

        def report():
            global gen
            best = gp.BestModel
            bestfit = best.GetFitness(0)
            sys.stdout.write('\r')
            cursor = int(np.round(gen / config.Generations * max_ticks))
            for i in range(cursor):
                sys.stdout.write('\u2588')
            sys.stdout.write(' ' * (max_ticks - cursor))
            sys.stdout.write(
                f'{100 * gen / config.Generations:.1f}%, generation {gen}/{config.Generations}, best_fit:{bestfit:.6f}, train quality:{-bestfit:.6f}, elapsed: {time.time() - t0:.2f}s')
            sys.stdout.flush()
            gen += 1

        gp.Run(rng, report, threads=self.threads_num)

        # get the best solution and print it
        best = gp.BestModel
        model_string = Operon.InfixFormatter.Format(best.Genotype, self.ds, 6)
        model_fit = best.GetFitness(0)
        self.model_string = model_string
        self.model_fit = model_fit
        print(f'\n{model_string}')
        # bingo_equ = trans_op0(model_string)
        # print(bingo_equ)
        # eval = SingleBingoEvaluator(self.data, bingo_equ)
        # model_fit = eval.do()
        print("最好适应度" + f'\n{model_fit}')
        self.best_fit = model_fit
        self.elapse_time = time.time() - t
        self.ind_list = gp.Individuals


class KeplarOperon(Alg):

    def __init__(self, max_generation, up_op_list, down_op_list, eval_op_list, error_tolerance, population, data
                 , operators, recursion_limit=10):
        super().__init__(max_generation, up_op_list, down_op_list, eval_op_list, error_tolerance, population)
        self.best_fit = None
        self.elapse_time = None
        self.operators = operators
        self.recursion_limit = recursion_limit
        self.data = data

    def run(self):
        t = time.time()
        dbscan = SklearnDBscan1()
        x, num = dbscan.do(self.data)
        db_sum = x
        n_cluster = num
        programs = []
        fit_list = []
        abRockSum = 0
        abRockNum = []
        epolusion = self.error_tolerance
        final_fit = 100
        recursion_limit = 10
        now_recursion = 1
        while epolusion < final_fit and now_recursion <= recursion_limit:
            for db_i in db_sum:
                print("数据shape" + str(np.shape(db_i)))
                data_i = Data("numpy", db_i, ["x1", "x2", "x3", "x4", 'y'])
                data_i.read_file()
                ds_x = data_i.get_np_x()
                ds_y = data_i.get_np_y()
                ds_xy = data_i.get_np_ds()
                ds_xy = Operon.Dataset(ds_xy)
                # taylor = TaylorJudge(data_i, "taylorgp")
                # jd = taylor.do()
                # if jd == "end":
                #     programs.append([taylor.program])
                #     fit_list.append([taylor.end_fitness])
                #     abRockNum.append(100000)
                #     abRockSum += 100000
                # else:
                generation = self.max_generation
                pop_size = self.population.pop_size
                abRockNum.append(generation * pop_size)
                abRockSum += generation * pop_size
                selector = OperonSelector(5)
                evaluator = OperonEvaluator("RMSE", ds_x, ds_y, 0.5, True, "Operon")
                crossover = OperonCrossover(ds_x, ds_y, "Operon")
                mutation = OperonMutation(1, 1, 1, 0.5, ds_x, ds_y, 10, 50, "balanced", "Operon")
                reinsert = OperonReinserter(None, "ReplaceWorst", 10, "Operon", ds_x, ds_y)
                op_up_list = [mutation, crossover]
                op_down_list = [reinsert]
                eva_list = [evaluator]
                op_alg = OperonAlg(1000, op_up_list, op_down_list, eva_list, selector, 1e-5, 128, 16, ds_x, ds_y)
                op_alg.run()
                op_top3 = op_alg.get_n_top()
                top_str_ind = []
                top_fit_list = []
                for i in op_top3:
                    top_str_ind.append(str(Operon.InfixFormatter.Format(i.Genotype, ds_xy, 6)))
                    top_fit_list.append(i.GetFitness(0))
                programs.append(top_str_ind)
                fit_list.append(top_fit_list)

            # print(programs)
            # print(fit_list)
            # if n_cluster > 1:
            #
            #     spare = KeplarSpareseRegression(n_cluster, programs, fit_list, self.data, 488)
            #     spare.do()
            #     rockBestFit = spare.rockBestFit
            #     final_equ = spare.final_str_ind
            #     final_equ = trans_op1(final_equ)
            #     single_eval = SingleBingoEvaluator(self.data, final_equ)
            #     sta = BingoStatistic(final_equ)
            #     sta.pos_do()
            #     final_fit = single_eval.do()
            #     print(f"第{now_recursion}轮" + "最好个体" + str(final_equ) + "适应度:" + str(final_fit))
            #     ucb = KeplarJudgeUCB(n_cluster, abRockSum, abRockNum, rockBestFit)
            #     max_ucb_index = ucb.pos_do()
            #     db_s = db_sum[max_ucb_index]
            #     db_sum = [db_s]
            #     programs = []
            #     fit_list = []
            #     abRockSum = 0
            #     abRockNum = []
            #     n_cluster = 1
            # else:
            best_i = 0
            best_j = 0
            for i in range(len(fit_list)):
                for j in range(len(fit_list[i])):
                    if fit_list[i][j] < final_fit:
                        final_fit = fit_list[i][j]
                        best_i = 1
                        best_j = j
            final_equ = programs[best_j][best_j]
            print(f"第{now_recursion}轮" + "适应度:" + str(final_fit) + "最好个体:" + str(final_equ))
            now_recursion += 1
        self.elapse_time = time.time() - t
        self.best_fit = final_fit


class KeplarMOperon(Alg):

    def __init__(self, max_generation, up_op_list, down_op_list, eval_op_list, error_tolerance, population, data
                 , operators, recursion_limit=10, top_n=3, p_noise=0.1):
        super().__init__(max_generation, up_op_list, down_op_list, eval_op_list, error_tolerance, population)
        self.p_noise = p_noise
        self.top_n = top_n
        self.best_fit = None
        self.elapse_time = None
        self.operators = operators
        self.recursion_limit = recursion_limit
        self.data = data

    def run(self):
        t = time.time()
        dbscan = SklearnDBscan(self.p_noise)
        x, num = dbscan.do(self.data)
        if not x:
            raise ValueError("数据集有问题")
        db_sum = x
        n_cluster = num
        programs = []
        fit_list = []
        op_list = []
        abRockSum = 0
        abRockNum = []
        for i in range(n_cluster):
            abRockNum.append(0)
        epolusion = self.error_tolerance
        final_fit = 100
        recursion_limit = 10
        now_recursion = 1
        taylor_flag = True
        first_flag = True
        selected_index = None
        while epolusion < final_fit and now_recursion <= recursion_limit:
            if first_flag or n_cluster == 1:
                for db_num, db_i in enumerate(db_sum):
                    print("数据shape" + str(np.shape(db_i)))
                    data_i = Data("numpy", db_i, ["x1", "x2", "x3", "x4", 'y'])
                    data_i.read_file()
                    ds_x = data_i.get_np_x()
                    ds_y = data_i.get_np_y()
                    ds_xy = data_i.get_np_ds()
                    ds_xy = Operon.Dataset(ds_xy)
                    taylor = TaylorJudge(data_i, "taylorgp")
                    if taylor_flag:
                        jd = taylor.do()
                    else:
                        jd = "not end"
                    if jd == "end":
                        programs.append([taylor.program])
                        fit_list.append([taylor.end_fitness])
                        abRockNum.append(100000)
                        abRockSum += 100000
                    else:
                        taylor_flag = False
                        generation = self.max_generation
                        pop_size = self.population.pop_size
                        abRockNum[db_num] += (generation * pop_size)
                        abRockSum += generation * pop_size
                        selector = OperonSelector(5)
                        evaluator = OperonEvaluator("RMSE", ds_x, ds_y, 0.5, True, "Operon")
                        crossover = OperonCrossover(ds_x, ds_y, "Operon")
                        mutation = OperonMutation(1, 1, 1, 0.5, ds_x, ds_y, 10, 50, "balanced", "Operon")
                        reinsert = OperonReinserter(None, "ReplaceWorst", 10, "Operon", ds_x, ds_y)
                        op_up_list = [mutation, crossover]
                        op_down_list = [reinsert]
                        eva_list = [evaluator]
                        x_shape = np.shape(ds_x[0])[0]
                        op_alg = OperonAlg(generation, op_up_list, op_down_list, eva_list, selector, 1e-5,
                                           pop_size, 16, ds_x, ds_y, data_i,x_shape)
                        op_alg.run()
                        op_top3 = op_alg.get_n_top(self.top_n)
                        top_str_ind = []
                        top_fit_list = []
                        for i in op_top3:
                            top_str_ind.append(str(Operon.InfixFormatter.Format(i.Genotype, ds_xy, 6)))
                            top_fit_list.append(i.GetFitness(0))
                        programs.append(top_str_ind)
                        fit_list.append(top_fit_list)
                        op_list.append(op_top3)
            else:
                db_i = db_sum[selected_index]
                print("数据shape" + str(np.shape(db_i)))
                data_i = Data("numpy", db_i, ["x1", "x2", "x3", "x4", 'y'])
                data_i.read_file()
                ds_x = data_i.get_np_x()
                ds_y = data_i.get_np_y()
                ds_xy = data_i.get_np_ds()
                ds_xy = Operon.Dataset(ds_xy)
                taylor = TaylorJudge(data_i, "taylorgp")
                if taylor_flag:
                    jd = taylor.do()
                else:
                    jd = "not end"
                if jd == "end":
                    programs.append([taylor.program])
                    fit_list.append([taylor.end_fitness])
                    abRockNum.append(100000)
                    abRockSum += 100000
                else:
                    taylor_flag = False
                    generation = self.max_generation
                    pop_size = self.population.pop_size
                    abRockNum[db_num] += (generation * pop_size)
                    abRockSum += generation * pop_size
                    selector = OperonSelector(5)
                    evaluator = OperonEvaluator("RMSE", ds_x, ds_y, 0.5, True, "Operon")
                    crossover = OperonCrossover(ds_x, ds_y, "Operon")
                    mutation = OperonMutation(1, 1, 1, 0.5, ds_x, ds_y, 10, 50, "balanced", "Operon")
                    reinsert = OperonReinserter(None, "ReplaceWorst", 10, "Operon", ds_x, ds_y)
                    op_up_list = [mutation, crossover]
                    op_down_list = [reinsert]
                    eva_list = [evaluator]
                    op_alg = OperonAlg(generation, op_up_list, op_down_list, eva_list, selector, 1e-5,
                                       pop_size, 16, ds_x, ds_y)
                    op_alg.run()
                    op_top_n = op_alg.get_n_top()
                    top_str_ind = []
                    top_fit_list = []
                    former_op_top_n = op_list[selected_index]
                    now_op_top_n = former_op_top_n + op_top_n
                    now_op_top_n = find_k_smallest_elements(now_op_top_n, self.top_n)
                    for i in now_op_top_n:
                        top_str_ind.append(str(Operon.InfixFormatter.Format(i.Genotype, ds_xy, 6)))
                        top_fit_list.append(i.GetFitness(0))
                    programs[selected_index] = top_str_ind
                    fit_list[selected_index] = top_fit_list
            # print(programs)
            # print(fit_list)
            if n_cluster > 1:
                spare = KeplarSpareseRegression(n_cluster, programs, fit_list, self.data, 488)
                spare.do()
                rockBestFit = spare.rockBestFit
                final_equ = spare.final_str_ind
                single_eval = SingleBingoEvaluator(self.data, final_equ)
                final_fit = single_eval.do()
                for num in range(len(fit_list)):
                    if fit_list[num][0] < final_fit:
                        final_fit = fit_list[num][0]
                        final_equ = programs[num][0]
                print(f"第{now_recursion}轮" + "适应度:" + str(final_fit) + "最佳个体" + str(final_equ))
                ucb = KeplarJudgeUCB(n_cluster, abRockSum, abRockNum, rockBestFit)
                max_ucb_index = ucb.pos_do()
                selected_index = max_ucb_index
            else:
                for i in range(len(fit_list)):
                    for j in range(len(fit_list[i])):
                        if fit_list[i][j] < final_fit:
                            final_fit = fit_list[i][j]
                            final_equ = programs[i][j]
                print(f"第{now_recursion}轮" + "适应度:" + str(final_fit) + "最佳个体" + str(final_equ))
            now_recursion += 1
        self.elapse_time = time.time() - t
        self.best_fit = final_fit


class KeplarMOperon2(Alg):

    def __init__(self, max_generation, up_op_list, down_op_list, eval_op_list, error_tolerance, population, data
                 , operators, recursion_limit=10, top_n=3, p_noise=0.1):
        super().__init__(max_generation, up_op_list, down_op_list, eval_op_list, error_tolerance, population)
        self.p_noise = p_noise
        self.top_n = top_n
        self.best_fit = None
        self.elapse_time = None
        self.operators = operators
        self.recursion_limit = recursion_limit
        self.data = data

    def run(self):
        t = time.time()
        dbscan = SklearnDBscan(self.p_noise)
        x, num = dbscan.do(self.data)
        if not x:
            raise ValueError("数据集有问题")
        db_sum = x
        n_cluster = num
        programs = []
        fit_list = []
        op_list = []
        abRockSum = 0
        abRockNum = []
        for i in range(n_cluster):
            abRockNum.append(0)
        epolusion = self.error_tolerance
        final_fit = 100
        recursion_limit = 10
        now_recursion = 1
        taylor_flag = True
        first_flag = True
        selected_index = None
        while epolusion < final_fit and now_recursion <= recursion_limit:
            if first_flag or n_cluster == 1:
                for db_num, db_i in enumerate(db_sum):
                    print("数据shape" + str(np.shape(db_i)))
                    data_i = Data("numpy", db_i, ["x1", "x2", "x3", "x4", 'y'])
                    data_i.read_file()
                    ds_x = data_i.get_np_x()
                    ds_y = data_i.get_np_y()
                    ds_xy = data_i.get_np_ds()
                    ds_xy = Operon.Dataset(ds_xy)
                    taylor = TaylorJudge(data_i, "taylorgp")
                    if taylor_flag:
                        jd = taylor.do()
                    else:
                        jd = "not end"
                    if jd == "end":
                        programs.append([taylor.program])
                        fit_list.append([taylor.end_fitness])
                        abRockNum.append(100000)
                        abRockSum += 100000
                    else:
                        taylor_flag = False
                        generation = self.max_generation
                        pop_size = self.population.pop_size
                        abRockNum[db_num] += (generation * pop_size)
                        abRockSum += generation * pop_size
                        selector = OperonSelector(5)
                        evaluator = OperonEvaluator("RMSE", ds_x, ds_y, 0.5, True, "Operon")
                        crossover = OperonCrossover(ds_x, ds_y, "Operon")
                        mutation = OperonMutation(1, 1, 1, 0.5, ds_x, ds_y, 10, 50, "balanced", "Operon")
                        reinsert = OperonReinserter(None, "ReplaceWorst", 10, "Operon", ds_x, ds_y)
                        op_up_list = [mutation, crossover]
                        op_down_list = [reinsert]
                        eva_list = [evaluator]
                        op_alg = OperonAlg(generation, op_up_list, op_down_list, eva_list, selector, 1e-5,
                                           pop_size, 16, ds_x, ds_y)
                        op_alg.run()
                        op_top3 = op_alg.get_n_top(self.top_n)
                        top_str_ind = []
                        top_fit_list = []
                        for i in op_top3:
                            top_str_ind.append(str(Operon.InfixFormatter.Format(i.Genotype, ds_xy, 6)))
                            top_fit_list.append(i.GetFitness(0))
                        programs.append(top_str_ind)
                        fit_list.append(top_fit_list)
                        op_list.append(op_top3)
            else:
                db_i = db_sum[selected_index]
                print("数据shape" + str(np.shape(db_i)))
                data_i = Data("numpy", db_i, ["x1", "x2", "x3", "x4", 'y'])
                data_i.read_file()
                ds_x = data_i.get_np_x()
                ds_y = data_i.get_np_y()
                ds_xy = data_i.get_np_ds()
                ds_xy = Operon.Dataset(ds_xy)
                taylor = TaylorJudge(data_i, "taylorgp")
                if taylor_flag:
                    jd = taylor.do()
                else:
                    jd = "not end"
                if jd == "end":
                    programs.append([taylor.program])
                    fit_list.append([taylor.end_fitness])
                    abRockNum.append(100000)
                    abRockSum += 100000
                else:
                    taylor_flag = False
                    generation = self.max_generation
                    pop_size = self.population.pop_size
                    abRockNum[db_num] += (generation * pop_size)
                    abRockSum += generation * pop_size
                    selector = OperonSelector(5)
                    evaluator = OperonEvaluator("RMSE", ds_x, ds_y, 0.5, True, "Operon")
                    crossover = OperonCrossover(ds_x, ds_y, "Operon")
                    mutation = OperonMutation(1, 1, 1, 0.5, ds_x, ds_y, 10, 50, "balanced", "Operon")
                    reinsert = OperonReinserter(None, "ReplaceWorst", 10, "Operon", ds_x, ds_y)
                    op_up_list = [mutation, crossover]
                    op_down_list = [reinsert]
                    eva_list = [evaluator]
                    op_alg = OperonAlg(generation, op_up_list, op_down_list, eva_list, selector, 1e-5,
                                       pop_size, 16, ds_x, ds_y)
                    op_alg.run()
                    op_top_n = op_alg.get_n_top()
                    top_str_ind = []
                    top_fit_list = []
                    former_op_top_n = op_list[selected_index]
                    now_op_top_n = former_op_top_n + op_top_n
                    now_op_top_n = find_k_smallest_elements(now_op_top_n, self.top_n)
                    for i in now_op_top_n:
                        top_str_ind.append(str(Operon.InfixFormatter.Format(i.Genotype, ds_xy, 6)))
                        top_fit_list.append(i.GetFitness(0))
                    programs[selected_index] = top_str_ind
                    fit_list[selected_index] = top_fit_list
            # print(programs)
            # print(fit_list)
            if n_cluster > 1:
                spare = KeplarSpareseRegression(n_cluster, programs, fit_list, self.data, 488)
                spare.do()
                rockBestFit = spare.rockBestFit
                final_equ = spare.final_str_ind
                single_eval = SingleBingoEvaluator(self.data, final_equ)
                final_fit = single_eval.do()
                for num in range(len(fit_list)):
                    if fit_list[num][0] < final_fit:
                        final_fit = fit_list[num][0]
                        final_equ = programs[num][0]
                print(f"第{now_recursion}轮" + "适应度:" + str(final_fit) + "最佳个体" + str(final_equ))
                ucb = KeplarJudgeUCB(n_cluster, abRockSum, abRockNum, rockBestFit)
                max_ucb_index = ucb.pos_do()
                selected_index = max_ucb_index
                pruning = MOperonPruning(fit_list, n_cluster, now_recursion, self.recursion_limit,programs,op_list,abRockSum,abRockNum)
                pruning.pos_do()
            else:
                for i in range(len(fit_list)):
                    for j in range(len(fit_list[i])):
                        if fit_list[i][j] < final_fit:
                            final_fit = fit_list[i][j]
                            final_equ = programs[i][j]
                print(f"第{now_recursion}轮" + "适应度:" + str(final_fit) + "最佳个体" + str(final_equ))
            now_recursion += 1
        self.elapse_time = time.time() - t
        self.best_fit = final_fit
