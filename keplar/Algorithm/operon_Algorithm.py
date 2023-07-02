import random
import sys
import time

import numpy as np

from keplar.Algorithm.Alg import Alg
from keplar.operator.creator import OperonCreator
import pyoperon as Operon

gen = 0
max_ticks = 50

t0 = time.time()


class OperonAlg(Alg):
    def __init__(self, max_generation, up_op_list, down_op_list, eval_op_list, sel,
                 error_tolerance, population_size, threads_num, np_x, np_y, minL=1, maxL=50, maxD=10,
                 decimalPrecision=5, population=None):
        super().__init__(max_generation, up_op_list, down_op_list, eval_op_list, error_tolerance, population)
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

    def run(self):
        selector = self.sel.do()
        target = self.ds.Variables[-1]
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
        evaluator.Budget = 10000 * 10000 # computational budget
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
                                               population_size=1000, pool_size=1000, p_crossover=1.0, p_mutation=0.25,
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
                f'{100 * gen / config.Generations:.1f}%, generation {gen}/{config.Generations}, train quality: {-bestfit:.6f}, elapsed: {time.time() - t0:.2f}s')
            sys.stdout.flush()
            gen += 1

        gp.Run(rng, report, threads=self.threads_num)

        # get the best solution and print it
        best = gp.BestModel
        model_string = Operon.InfixFormatter.Format(best.Genotype, self.ds, 6)
        print(f'\n{model_string}')
