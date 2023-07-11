import numpy as np

from keplar.Algorithm.Alg import BingoAlg
from keplar.Algorithm.operon_Algorithm import OperonAlg
from keplar.data.data import Data
from keplar.operator.JudgeUCB import KeplarJudgeUCB
from keplar.operator.crossover import OperonCrossover
from keplar.operator.evaluator import OperonEvaluator, MetricsBingoEvaluator
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
for i, db_i in enumerate(db_sum):
    data_i = Data("numpy", db_i, ["x1", "x2", "x3", "x4", 'y'])
    data_i.read_file()
    taylor = TaylorJudge(data_i, "taylorgp")
    jd = taylor.do()
    if jd == "end":
        programs.append([taylor.program])
        fit_list.append([taylor.end_fitness])
        abRockNum[i] = 100000
        abRockSum += 100000
    else:
        generation = 1000
        pop_size = 128
        abRockNum[i] += generation * pop_size
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
    ucb=KeplarJudgeUCB(n_cluster,abRockSum,abRockNum)
    ucb.do()

