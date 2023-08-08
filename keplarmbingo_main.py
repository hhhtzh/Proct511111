import numpy as np
import pandas as pd

from keplar.Algorithm.Alg import BingoAlg, KeplarMBingo
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
from keplar.population.population import Population
from keplar.preoperator.ml.sklearndbscan import SklearnDBscan

# data = Data("txt", "datasets/1.txt", ["x", "y"])
# data.read_file()
# data.set_xy("y")
fit_list = []
time_list = []
# data = Data("pmlb", "1027_ESL", ["x1", "x2", "x3", "x4", 'y'])
data = Data("txt", "datasets/vla/two/1.txt", ["x0", "x1", "y"])
# data = Data("txt", "datasets/2.txt", ["x0", "x1", "x2", "x3", "x4", "y"])
data.read_file()
data.set_xy("y")
# data.read_file()
operators = ["+", "-", "*", "/", "sin", "exp", "cos", 'sqrt', 'log', 'sin', 'pow', 'exp', '^']
population = Population(128)
kmb = KeplarMBingo(1000, None, None, None, 0.1, population, data, operators)
for i in range(10):
    kmb.run()
    fit_list.append(kmb.best_fit)
    time_list.append(kmb.elapse_time)
fit_pd = pd.DataFrame({'KeplarMBingo': fit_list})
time_pd = pd.DataFrame({'KeplarMBingo': time_list})
# fit_pd.to_csv(r"result/vla_5.csv", sep=',', mode="a")
# time_pd.to_csv(r"result/vla_5_time.csv", sep=',', mode="a")
fit_pd.to_csv(r"result/vla_5.csv", sep=',', mode="a")
time_pd.to_csv(r"result/vla_5_time.csv", sep=',', mode="a")
