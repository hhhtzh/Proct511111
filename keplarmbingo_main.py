import numpy as np

from keplar.Algorithm.Alg import BingoAlg
from keplar.Algorithm.operon_Algorithm import OperonAlg
from keplar.data.data import Data
from keplar.operator.JudgeUCB import KeplarJudgeUCB
from keplar.operator.crossover import OperonCrossover
from keplar.operator.evaluator import OperonEvaluator, MetricsBingoEvaluator, SingleBingoEvaluator
from keplar.operator.mutation import OperonMutation
from keplar.operator.reinserter import OperonReinserter
from keplar.operator.selector import OperonSelector
from keplar.operator.sparseregression import KeplarSpareseRegression
from keplar.operator.taylor_judge import TaylorJudge
from keplar.preoperator.ml.sklearndbscan import SklearnDBscan

# data = Data("txt", "datasets/1.txt", ["x", "y"])
# data.read_file()
# data.set_xy("y")
data = Data("pmlb", "1027_ESL", ["x1", "x2", "x3", "x4", 'y'])
data.read_file()
operators = ["+", "-", "*", "/", "sin", "exp"]
for i in [1e-5, 0.2, 1, 4, 10, 100]:
    dbscan = SklearnDBscan(eps=i)
    x, num = dbscan.do(data)
    if x:
        break

db_sum = x
n_cluster = num
programs = []
fit_list = []
top3s = []
abRockSum = 0
abRockNum = []
epolusion = 0.1
final_fit = 100
recursion_limit = 10
now_recursion = 0
while epolusion < final_fit and now_recursion < recursion_limit:
    for db_i in db_sum:
        print("数据shape"+str(np.shape(db_i)))
        data_i = Data("numpy", db_i, ["x1", "x2", "x3", "x4", 'y'])
        data_i.read_file()
        taylor = TaylorJudge(data_i, "taylorgp")
        jd = taylor.do()
        if jd == "end":
            programs.append([taylor.program])
            fit_list.append([taylor.end_fitness])
            abRockNum.append(100000)
            abRockSum += 100000
        else:
            generation = 1000
            pop_size = 128
            abRockNum.append(generation * pop_size)
            abRockSum += generation * pop_size
            bingo = BingoAlg(generation, data, operators=operators, POP_SIZE=pop_size)
            bingo.run()
            bingo_top3 = bingo.island.get3top()
            top_str_ind = []
            top_fit_list = []
            for i in bingo_top3:
                top_str_ind.append(str(i))
                top_fit_list.append(i.fitness)
            programs.append(top_str_ind)
            fit_list.append(top_fit_list)

    # print(programs)
    # print(fit_list)
    if n_cluster > 1:
        spare = KeplarSpareseRegression(n_cluster, programs, fit_list, data, 488)
        spare.do()
        final_best_fit = spare.bestLassoFitness
        rockBestFit = spare.rockBestFit
        final_equ = spare.final_str_ind
        single_eval = SingleBingoEvaluator(data, final_equ)
        final_fit = single_eval.do()
        print(f"第{now_recursion}轮" + "适应度:" + str(final_fit))
        ucb = KeplarJudgeUCB(n_cluster, abRockSum, abRockNum, rockBestFit)
        max_ucb_index = ucb.pos_do()
        db_s=db_sum[max_ucb_index]
        db_sum = [db_s]
        programs = []
        fit_list = []
        top3s = []
        abRockSum = 0
        abRockNum = []
        n_cluster=1
    else:
        print(f"第{now_recursion}轮" + "适应度:" + str(final_fit))
    now_recursion += 1
