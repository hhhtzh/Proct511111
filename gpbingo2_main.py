import pandas as pd
from pmlb import fetch_data
from keplar.Algorithm.Alg import KeplarBingoAlg, GpBingoAlg, GpBingo2Alg

from keplar.data.data import Data
from keplar.operator.cleaner import BingoCleaner
from keplar.operator.composite_operator import CompositeOp, CompositeOpReturn
from keplar.operator.creator import BingoCreator, GpCreator
from keplar.operator.crossover import BingoCrossover
from keplar.operator.evaluator import BingoEvaluator, GpEvaluator
from keplar.operator.generator import BingoGenerator
from keplar.operator.mutation import BingoMutation
from keplar.operator.selector import BingoSelector

# data = Data("txt", "datasets/1.txt", ["x", "y"])
data = Data("txt", "datasets/vla/two/1.txt", ["x0", "x1", "y"])
# data = Data("txt", "datasets/2.txt", ["x0", "x1", "x2", "x3", "x4", "y"])
# data = Data("pmlb", "1027_ESL", ["x1", "x2", "x3", 'y'])
data.read_file()
data.set_xy("y")
x = data.get_np_x()
y = data.get_np_y()
fit_list = []
time_list = []
operators = ["+", "-", "*", "/", "sin", "exp", "cos", 'sqrt', 'log', 'sin', 'pow', 'exp', '^']
creator = GpCreator(128, x, y, "gplearn", n_jobs=20)
evaluator = GpEvaluator(x, y, "Bingo", metric="rmse")
crossover = BingoCrossover("Bingo")
mutation = BingoMutation(x, operators, "Bingo")
selector = BingoSelector(0.5, "tournament", "Bingo")
gen_up_oplist = CompositeOp([crossover, mutation])
gen_down_oplist = CompositeOpReturn([selector])
gen_eva_oplist = CompositeOp([evaluator])
# for i in range(10):
for i in range(10):
    population = creator.do()
    bgsr = GpBingo2Alg(1000, gen_up_oplist, gen_down_oplist, gen_eva_oplist, 0.001, population)
    bgsr.run()
    fit_list.append(bgsr.best_fit)
    time_list.append(bgsr.elapse_time)
# fit_list.append(bgsr.best_fit)
# time_list.append(bgsr.elapse_time)
fit_pd = pd.DataFrame({'GpBingo2': fit_list})
time_pd = pd.DataFrame({'GpBingo2': time_list})
fit_pd.to_csv(r"result/vla_2_1.csv", sep=',', mode="a")
time_pd.to_csv(r"result/vla_2_1_time.csv", sep=',', mode="a")
