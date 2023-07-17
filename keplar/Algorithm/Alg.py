import random
from abc import abstractmethod
import time

import numpy as np

from bingo.evaluation.evaluation import Evaluation
from bingo.evolutionary_algorithms.age_fitness import AgeFitnessEA
from bingo.evolutionary_optimizers.island import Island
from bingo.local_optimizers.local_opt_fitness import LocalOptFitnessFunction
from bingo.local_optimizers.scipy_optimizer import ScipyOptimizer
from bingo.symbolic_regression import ExplicitTrainingData, ComponentGenerator, AGraphCrossover, AGraphMutation, \
    ExplicitRegression, AGraphGenerator
from keplar.operator.composite_operator import CompositeOp
# from keplar.operator.reinserter import OperonReinserter, KeplarReinserter
import pyoperon as Operon

from keplar.operator.reinserter import KeplarReinserter
from keplar.population.newisland import NewIsland
from keplar.preoperator.ml.sklearndbscan import SklearnDBscan


# class SR_Alg(CompositeOp):
#     def __init__(self,Comop_list):
#         super().__init__()
#         self.Comop_list = Comop_list

#     def do(self):
#         for Comop in self.Comop_list:
#             Comop.do()

class Alg:
    def __init__(self, max_generation, up_op_list, down_op_list, eval_op_list, error_tolerance, population):
        self.max_generation = max_generation
        self.up_op_list = up_op_list
        self.down_op_list = down_op_list
        self.eval_op_list = eval_op_list
        self.error_tolerance = error_tolerance
        self.population = population
        self.age = 0

    @abstractmethod
    def run(self):
        raise NotImplementedError


class uDSR_Alg(Alg):
    def __init__(self, max_generation, up_op_list, down_op_list, eva_op_list, error_tolerance, population):
        super().__init__(max_generation, up_op_list, down_op_list, eva_op_list, error_tolerance, population)


class KeplarBingoAlg(Alg):

    def __init__(self, max_generation, up_op_list, down_op_list, eva_op_list, error_tolerance, population):
        super().__init__(max_generation, up_op_list, down_op_list,
                         eva_op_list, error_tolerance, population)
        self.elapse_time = None
        self.best_fit = None

    def get_best_individual(self):
        return self.population.target_pop_list[self.population.get_tar_best()]

    def run(self):
        t = time.time()
        generation_pop_size = self.population.get_pop_size()
        self.eval_op_list.do(self.population)
        now_error = self.population.get_best_fitness()
        while self.age < self.max_generation and now_error >= self.error_tolerance or str(now_error) == "nan":
            self.population = self.down_op_list.do(self.population)
            while generation_pop_size > self.population.get_pop_size():
                self.up_op_list.do(self.population)
            self.eval_op_list.do(self.population)
            now_error = self.population.get_best_fitness()
            # best_ind = str(self.get_best_individual())
            self.age += 1
            # print("第" + f"{self.age}代种群，" +
            #       f"最佳个体适应度为{now_error}" + f"最佳个体为{best_ind}")
        best_ind = str(self.get_best_individual())
        print("迭代结束，共迭代" + f"{self.age}代" +
              f"最佳个体适应度为{now_error}" + f"最佳个体为{best_ind}")
        self.best_fit = now_error
        self.elapse_time = time.time() - t


class TaylorBingoAlg(Alg):
    def __init__(self, max_generation, up_op_list, down_op_list, eva_op_list, error_tolerance, population, fe_list):
        super().__init__(max_generation, up_op_list, down_op_list,
                         eva_op_list, error_tolerance, population)
        self.elapse_time = None
        self.best_fit = None
        self.fe_list = fe_list

    def get_best_individual(self):
        return self.population.target_pop_list[self.population.get_tar_best()]

    def run(self):
        t = time.time()
        for i in self.fe_list:
            result = i.do()
            if result == "end":
                print("使用泰勒GP回归结束，共迭代" + "0代" +
                      f"最佳个体适应度为{i.end_fitness}," + f"最佳个体为{i.program}")
                return 1
        generation_pop_size = self.population.get_pop_size()
        self.eval_op_list.do(self.population)
        now_error = self.population.get_best_fitness()
        while self.age < self.max_generation and now_error >= self.error_tolerance or str(now_error) == "nan":
            self.population = self.down_op_list.do(self.population)
            while generation_pop_size > self.population.get_pop_size():
                self.up_op_list.do(self.population)
            self.eval_op_list.do(self.population)
            now_error = self.population.get_best_fitness()
            best_ind = str(self.get_best_individual())
            self.age += 1
            print("第" + f"{self.age}代种群，" +
                  f"最佳个体适应度为{now_error}" + f"最佳个体为{best_ind}")
        best_ind = str(self.get_best_individual())
        print("迭代结束，共迭代" + f"{self.age}代" +
              f"最佳个体适应度为{now_error}" + f"最佳个体为{best_ind}")
        self.best_fit = now_error
        self.elapse_time = time.time() - t


class OperonBingoAlg(Alg):

    def __init__(self, max_generation, up_op_list, down_op_list, eval_op_list, error_tolerance, population, selector,
                 np_x, np_y, pool_size):
        super().__init__(max_generation, up_op_list, down_op_list, eval_op_list, error_tolerance, population)
        self.elapse_time = None
        self.best_fit = None
        self.pool_size = pool_size
        self.selector = selector
        self.np_y = np_y.reshape([-1, 1])
        self.np_x = np_x
        self.ds = Operon.Dataset(np.hstack([np_x, self.np_y]))

    def get_best_individual(self):
        if self.population.pop_type != "self":
            best_num = self.population.get_tar_best()
            str_op_tree = Operon.InfixFormatter.Format(self.population.target_pop_list[best_num], self.ds, 5)
            return str_op_tree
        else:
            best_num = self.population.get_best()
            return self.population.pop_list[best_num].format()

    def run(self):
        t=time.time()
        for i in self.eval_op_list:
            i.do(self.population)
        now_error = self.population.get_best_fitness()
        while self.age < self.max_generation and now_error >= self.error_tolerance or str(now_error) == "nan":
            pool_list = self.selector.do(self.population)
            while len(pool_list.target_pop_list) < self.pool_size:
                for i in self.up_op_list:
                    i.do(pool_list)
            for i in self.eval_op_list:
                i.do(pool_list)
            reinserter = KeplarReinserter(pool_list, "self")
            reinserter.do(self.population)
            now_error = self.population.get_best_fitness()
            # best_ind = str(self.get_best_individual())
            self.age += 1
            # print("第" + f"{self.age}代种群，" +
            #       f"最佳个体适应度为{now_error}" + f"最佳个体为{best_ind}")
        best_ind = str(self.get_best_individual())

        print("迭代结束，共迭代" + f"{self.age}代" +
              f"最佳个体适应度为{now_error}" + f"最佳个体为{best_ind}")
        self.best_fit = now_error
        self.elapse_time = time.time() - t


class GplearnAlg(Alg):
    def __init__(self, max_generation, up_op_list, down_op_list, eval_op_list, error_tolerance, population):
        super().__init__(max_generation, up_op_list, down_op_list, eval_op_list, error_tolerance, population)

    def run(self):
        for i in self.eval_op_list:
            i.do(self.population)
        now_error = self.population.get_best_fitness()
        while self.age < self.max_generation and now_error >= self.error_tolerance or str(now_error) == "nan":
            pool_list = self.selector.do(self.population)
            while len(pool_list.target_pop_list) < self.pool_size:
                for i in self.up_op_list:
                    i.do(pool_list)
            for i in self.eval_op_list:
                i.do(pool_list)
            reinserter = KeplarReinserter(pool_list, "self")
            reinserter.do(self.population)
            now_error = self.population.get_best_fitness()
            best_ind = str(self.get_best_individual())
            self.age += 1
            print("第" + f"{self.age}代种群，" +
                  f"最佳个体适应度为{now_error}" + f"最佳个体为{best_ind}")
        best_ind = str(self.get_best_individual())

        print("迭代结束，共迭代" + f"{self.age}代" +
              f"最佳个体适应度为{now_error}" + f"最佳个体为{best_ind}")


class TaylorGpAlg(Alg):
    def __init__(self, max_generation, up_op_list, down_op_list, eval_op_list, error_tolerance, population):
        super().__init__(max_generation, up_op_list, down_op_list, eval_op_list, error_tolerance, population)

    def run(self):
        pass
        # x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24, x25, x26, x27, x28, x29 = symbols(
        #     "x0,x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16,x17,x18,x19,x20,x21,x22,x23,x24,x25,x26,x27,x28,x29 ")
        # _x = [x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22,
        #       x23, x24, x25, x26, x27, x28, x29]
        # eqName = r'eq_' + str(self.output_fileName) + '.csv'
        # # eqName = r'result/eq_' + str(fileNum) + '.csv'
        # eq_write = open(eqName, "w+")  # 重新写
        # eq_write.write(
        #     'Gen|Top1|Length1|Fitness1|Top2|Length2|Fitness2|Top3|Length3|Fitness3|Top4|Length4|Fitness4|Top5|Length5'
        #     '|Fitness5|Top6|Length6|Fitness6|Top7|Length7|Fitness7|Top8|Length8|Fitness8|Top9|Length9|Fitness9|Top10'
        #     '|Length10|Fitness10\n')
        # print('eqName=', self.output_fileName)
        # average_fitness = 0
        # repeat = 1
        # time_start1 = time.time()
        # for repeatNum in range(repeat):
        #     time_start2 = time.time()
        #     X = self.data.get_np_x()
        #     Y = self.data.get_np_y()
        #     np_ds = self.data.get_np_ds()
        #     loopNum = 0
        #     Metric = []
        #     while True:  # 进行5次低阶多项式的判别--不是则继续
        #         metric = Metrics(varNum=X.shape[1], dataSet=np_ds)
        #         loopNum += 1
        #         Metric.append(metric)
        #         if loopNum == 2 and X.shape[1] <= 2:
        #             break
        #         elif loopNum == 5 and (2 < X.shape[1] <= 3):
        #             break
        #         elif loopNum == 4 and (3 < X.shape[1] <= 4):
        #             break
        #         elif loopNum == 3 and (4 < X.shape[1] <= 5):
        #             break
        #         elif loopNum == 2 and (5 < X.shape[1] <= 6):
        #             break
        #         elif loopNum == 1 and (X.shape[1] > 6):
        #             break
        #     Metric.sort(key=lambda x: x.nmse)  # 按低阶多项式拟合误差对20个metric排序
        #     self.metric = Metric[0]
        #     print('排序后选中多项式_and_低阶多项式MSE:', self.metric.nmse, self.metric.low_nmse)
        #     eq_write.write(
        #         str(-1) + '|' + str(self.metric.f_low_taylor) + '|' + '10' + '|' + str(
        #             self.metric.low_nmse) + '|' + '\n')
        #     if self.metric.nmse < 0.1:
        #         self.metric.nihe_flag = True
        #     else:  # 拟合失败后Taylor特征的判别以数据集为准
        #         self.metric.bias = 0.
        #         print('拟合失败')
        #     Pop = 1000
        #     end_fitness, program = None, None
        #     # 下面分成有可分性的递归和非递归处理方式---对应需要Metric和Metric2两个类分别处理,Metric中的_x都是默认的
        #     if metric.judge_Low_polynomial():
        #         self.end_fitness, self.program = metric.low_nmse, metric.f_low_taylor
        #         return "end"
        #     elif metric.nihe_flag and (metric.judge_additi_separability() or metric.judge_multi_separability()):
        #         self.end_fitness, self.program = CalTaylorFeatures(metric.f_taylor, _x[:X.shape[1]], X, Y, Pop,
        #                                                            repeatNum,
        #                                                            eq_write)
        #         # nihe_flag, low_polynomial, qualified_list, Y = cal_Taylor_features(varNum=X.shape[1], fileNum=fileNum, Y=Y)
        #     else:  # 针对多变量不能拟合：仅根据数据集获取边界以及单调性指导SR--放在Taylor特征判别处理
        #
        #         qualified_list = []
        #         qualified_list.extend(
        #             [metric.judge_Bound(),
        #              metric.f_low_taylor,
        #              metric.low_mse,
        #              metric.bias,
        #              metric.judge_parity(),
        #              metric.judge_monotonicity()])
        #         print(qualified_list)
        #         end_fitness, program = Taylor_Based_SR(_x, X, metric.change_Y(Y), qualified_list, eq_write, Pop,
        #                                                repeatNum, metric.low_mse < 1e-5)
        #     print('fitness_and_program', end_fitness, program, sep=' ')
        #     average_fitness += end_fitness
        #     time_end2 = time.time()
        #     print('current_time_cost', (time_end2 - time_start2) / 3600, 'hour')


class GpBingoAlg(Alg):

    def __init__(self, max_generation, up_op_list, down_op_list, eva_op_list, error_tolerance, population):
        super().__init__(max_generation, up_op_list, down_op_list,
                         eva_op_list, error_tolerance, population)
        self.best_fit = None
        self.elapse_time = None

    def get_best_individual(self):
        return self.population.target_pop_list[self.population.get_tar_best()]

    def run(self):
        t = time.time()
        generation_pop_size = self.population.get_pop_size()
        self.eval_op_list.do(self.population)
        now_error = self.population.get_best_fitness()
        while self.age < self.max_generation and now_error >= self.error_tolerance or str(now_error) == "nan":
            self.population = self.down_op_list.do(self.population)
            while generation_pop_size > self.population.get_pop_size():
                self.up_op_list.do(self.population)
            self.eval_op_list.do(self.population)
            now_error = self.population.get_best_fitness()
            # best_ind = str(self.get_best_individual())
            self.age += 1
            # print("第" + f"{self.age}代种群，" +
            #       f"最佳个体适应度为{now_error}" + f"最佳个体为{best_ind}")
        best_ind = str(self.get_best_individual())
        print("迭代结束，共迭代" + f"{self.age}代" +
              f"最佳个体适应度为{now_error}" + f"最佳个体为{best_ind}")
        self.best_fit=now_error
        self.elapse_time = time.time() - t


class BingoAlg(Alg):

    def __init__(self, max_generation, data, operators, POP_SIZE=128,
                 STACK_SIZE=10,
                 MUTATION_PROBABILITY=0.4,
                 CROSSOVER_PROBABILITY=0.4,
                 NUM_POINTS=100,
                 START=-10,
                 STOP=10,
                 ERROR_TOLERANCE=1e-6, metric="rmse", up_op_list=None, down_op_list=None, eval_op_list=None,
                 error_tolerance=None,
                 population=None):
        super().__init__(max_generation, up_op_list, down_op_list, eval_op_list, error_tolerance, population)
        self.elapse_time = None
        self.best_fit = None
        self.best_ind = None
        self.island = None
        self.metric = metric
        self.ERROR_TOLERANCE = ERROR_TOLERANCE
        self.STOP = STOP
        self.START = START
        self.NUM_POINTS = NUM_POINTS
        self.CROSSOVER_PROBABILITY = CROSSOVER_PROBABILITY
        self.MUTATION_PROBABILITY = MUTATION_PROBABILITY
        self.STACK_SIZE = STACK_SIZE
        self.POP_SIZE = POP_SIZE
        self.operators = operators
        self.data = data

    def report_island_status(self, test_island):
        print("-----  Generation %d  -----" % test_island.generational_age)
        print("Best individual:     ", test_island.get_best_individual())
        print("Best fitness:        ", test_island.get_best_fitness())
        print("Fitness evaluations: ", test_island.get_fitness_evaluation_count())

    def init_island(self):
        np.random.seed(4)

        training_data = ExplicitTrainingData(self.data.get_np_x(), self.data.get_np_y())

        component_generator = ComponentGenerator(self.data.get_np_x().shape[1])
        for i in self.operators:
            component_generator.add_operator(i)

        crossover = AGraphCrossover()
        mutation = AGraphMutation(component_generator)

        agraph_generator = AGraphGenerator(self.STACK_SIZE, component_generator)

        fitness = ExplicitRegression(training_data=training_data, metric=self.metric)
        optimizer = ScipyOptimizer(fitness, method="lm")
        local_opt_fitness = LocalOptFitnessFunction(fitness, optimizer)
        evaluator = Evaluation(local_opt_fitness)

        ea = AgeFitnessEA(
            evaluator,
            agraph_generator,
            crossover,
            mutation,
            self.MUTATION_PROBABILITY,
            self.CROSSOVER_PROBABILITY,
            self.POP_SIZE,
        )

        island = NewIsland(ea, agraph_generator, self.POP_SIZE)
        return island

    def run(self):
        t = time.time()
        test_island = self.init_island()
        self.report_island_status(test_island)
        test_island.evolve_until_convergence(
            max_generations=1000, fitness_threshold=self.ERROR_TOLERANCE
        )
        self.report_island_status(test_island)
        self.island = test_island
        self.best_ind = test_island.get_best_individual()
        self.best_fit = test_island.get_best_fitness()
        self.elapse_time = time.time() - t


class KeplarMBingo(Alg):
    def __init__(self, max_generation, up_op_list, down_op_list, eval_op_list, error_tolerance, population,data):
        super().__init__(max_generation, up_op_list, down_op_list, eval_op_list, error_tolerance, population)
        self.data = data

    def do(self,population):
        for i in [1e-5, 0.2, 1, 4, 10, 100]:
            dbscan = SklearnDBscan(eps=i)
            x, num = dbscan.do(self.data)
            if x:
                break