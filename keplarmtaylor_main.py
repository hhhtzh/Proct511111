import numpy as np

from keplar.Algorithm.Alg import BingoAlg
from keplar.Algorithm.operon_Algorithm import OperonAlg
from keplar.data.data import Data
from keplar.operator.crossover import OperonCrossover
from keplar.operator.evaluator import OperonEvaluator
from keplar.operator.mutation import OperonMutation
from keplar.operator.reinserter import OperonReinserter
from keplar.operator.selector import OperonSelector
from keplar.operator.taylor_judge import TaylorJudge
from keplar.preoperator.ml.sklearndbscan import SklearnDBscan

data = Data("txt", "datasets/1.txt", ["x", "y"])
data.read_file()
data.set_xy("y")

for i in [1e-5, 0.2, 1, 4, 10, 100]:
    dbscan = SklearnDBscan(eps=i)
    x, num = dbscan.do(data)
    if x:
        break
db_sum = x
n_cluster = num
programs = []
fit_list = []
for db_i in db_sum:
    data_i = Data("numpy", db_i, ["x", "y"])
    data_i.read_file()
    taylor = TaylorJudge(data_i, "taylorgp")
    jd = taylor.do()
    if jd == "end":
        programs.append(taylor.program)
        fit_list.append(taylor.end_fitness)
    else:
        bingo = BingoAlg(1000, data, operators=["+", "-", "*", "/", "sin", "exp"])
        bingo.run()
        graph3 = bingo.island.get3top()
if n_cluster > 1:
    mean_fit = np.mean(fit_list)
    for i in range(len(fit_list)):
        if fit_list[i] > mean_fit:
            fit_list.pop(i)
            programs.pop(i)
