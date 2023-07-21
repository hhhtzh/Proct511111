import numpy as np

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
data = Data("pmlb", "1027_ESL", ["x1", "x2", "x3", "x4", 'y'])
data.read_file()
operators = ["+", "-", "*", "/", "sin", "exp", "sqrt", "^"]
population = Population(128)
kmb = KeplarMBingo(1000, None, None, None, 0.1, population, data, operators)
kmb.run()
