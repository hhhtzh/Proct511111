import random
import time

import numpy as np

from TaylorGP.TaylorGP import CalTaylorFeatures
from TaylorGP.src.taylorGP.calTaylor import Metrics
from bingo.symbolic_regression.agraph.string_parsing import postfix_to_command_array_and_constants
from gplearn.genetic import SymbolicRegressor

from bingo.symbolic_regression import ComponentGenerator, AGraphGenerator, AGraph
from keplar.population.individual import Individual
from keplar.operator.operator import Operator

from keplar.population.population import Population
from keplar.translator.translator import trans_gp, trans_op
import pyoperon as Operon


class Creator(Operator):
    def __init__(self):
        super().__init__()

    def do(self, population):
        pass


class BingoCreator(Creator):
    def __init__(self, pop_size, operators, x, stack_size, to_type):
        super().__init__()
        self.to_type = to_type
        self.pop_size = pop_size
        self.operators = operators
        self.population = []
        self.x = x
        self.stack_size = stack_size

    def do(self, population=None):
        component_generator = ComponentGenerator(self.x.shape[1])
        for op in self.operators:
            component_generator.add_operator(op)
        agraph_generator = AGraphGenerator(self.stack_size, component_generator)
        self.population = Population(pop_size=self.pop_size)
        pop = [agraph_generator() for _ in range(self.pop_size)]
        pop_list = []
        if self.to_type == "Bingo":
            self.population.target_pop_list = pop
            self.population.pop_type = "Bingo"
        else:
            pass
        return self.population


class GpCreator(Creator):
    def __init__(self, pop_size, x, y):
        super().__init__()
        self.pop_size = pop_size
        self.x = x
        self.y = y

    def do(self, population=None):
        pop = Population(self.pop_size)
        for i in range(self.pop_size):
            reg = SymbolicRegressor(generations=1, population_size=1)
            reg.fit(self.x, self.y)
            equ = trans_gp(str(reg))
            ind = Individual(str(equ))
            pop.append(ind)
        return pop


class OperonCreator(Creator):
    def __init__(self, tree_type, np_x, np_y, pop_size, to_type, minL=1, maxL=50, maxD=10, decimalPrecision=5):
        super().__init__()
        if tree_type == "balanced" or tree_type == "probabilistic":
            self.tree_type = tree_type
        else:
            raise ValueError("创建树的类型错误")
        self.maxD = maxD
        self.minL = minL
        self.maxL = maxL
        self.to_type = to_type
        self.pop_size = pop_size
        self.decimalPrecision = decimalPrecision
        np_y = np_y.reshape([-1, 1])
        self.ds = Operon.Dataset(np.hstack([np_x, np_y]))
        self.target = self.ds.Variables[-1]
        self.inputs = Operon.VariableCollection(v for v in self.ds.Variables if v.Name != self.target.Name)

    def do(self, population=None):
        pset = Operon.PrimitiveSet()
        pset.SetConfig(Operon.PrimitiveSet.TypeCoherent)
        if self.tree_type == "balanced":
            tree_creator = Operon.BalancedTreeCreator(pset, self.inputs, bias=0.0)
        elif self.tree_type == "probabilistic":
            tree_creator = Operon.ProbabilisticTreeCreator(pset, self.inputs, bias=0.0)
        else:
            raise ValueError("Operon创建树的类型名称错误")
        tree_initializer = Operon.UniformLengthTreeInitializer(tree_creator)
        tree_initializer.ParameterizeDistribution(self.minL, self.maxL)
        tree_initializer.MaxDepth = self.maxD
        rng = Operon.RomuTrio(random.randint(1, 1000000))
        coeff_initializer = Operon.NormalCoefficientInitializer()
        coeff_initializer.ParameterizeDistribution(0, 1)
        tree_list = []
        pop = Population(self.pop_size)
        pop.pop_type = "Operon"

        variable_list = self.ds.Variables
        for i in range(self.pop_size):
            tree = tree_initializer(rng)
            coeff_initializer(rng, tree)
            tree_list.append(tree)
        pop.check_flag(self.to_type)
        trans_flag = pop.translate_flag
        pop.target_pop_list = tree_list
        if trans_flag:
            for i in tree_list:
                func = trans_op(i, variable_list)
                ind_new = Individual(func=func)
                pop.append(ind_new)
                pop.self_pop_enable = True
        return pop


class uDSR_Creator(Creator):
    def __init__(self, T):
        super().__init__()
        self.T = T
        self.population = None

    def do(self, population=None):
        if self.T is None:
            print("T is None")
        else:
            length_T = len(self.T)
            # l = np.array([len(T[i]) for i in range(length_T)])
            self.population = Population(length_T)
            self.population.pop_type = "uDSR"
            self.population.target_pop_list = self.T
            return self.population


class TaylorGPCreator(Creator):
    def __init__(self, data, output_fileName):
        super().__init__()
        self.data = data
        self.program = None
        self.end_fitness = None
        self.metric = None
        self.output_fileName = output_fileName

    def do(self, population=None):
        pass
        # x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24, x25, x26, x27, x28, x29 = symbols(
        #     "x0,x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16,x17,x18,x19,x20,x21,x22,x23,x24,x25,x26,x27,"
        #     "x28,x29 ")
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
        #     if metric.judge_Low_polynomial():
        #         self.end_fitness, self.program = metric.low_nmse, metric.f_low_taylor
        #         return "end"
        #     elif metric.nihe_flag and (metric.judge_additi_separability() or metric.judge_multi_separability()):
        #         self.end_fitness, self.program = CalTaylorFeatures(metric.f_taylor, _x[:X.shape[1]], X, Y, Pop,
        #                                                            repeatNum,
        #                                                            eq_write)
        #         return "end"
        #         # nihe_flag, low_polynomial, qualified_list, Y = cal_Taylor_features(varNum=X.shape[1], fileNum=fileNum, Y=Y)
        #     else:  # 针对多变量不能拟合：仅根据数据集获取边界以及单调性指导SR--放在Taylor特征判别处理
        #
        #         qualified_list = []
        #         qualified_list.extend(
        #             [metric.judge_Bound(),
        #              metric.f_low_taylor,
        #              metric.low_nmse,
        #              metric.bias,
        #              metric.judge_parity(),
        #              metric.judge_monotonicity()])
        #         print(qualified_list)
        #         end_fitness, program = Taylor_Based_SR(_x, X, metric.change_Y(Y), qualified_list, eq_write, Pop,
        #                                                repeatNum, metric.low_nmse < 1e-5)
