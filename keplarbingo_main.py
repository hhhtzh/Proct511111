import pandas as pd
from pmlb import fetch_data
from keplar.Algorithm.Alg import KeplarBingoAlg

from keplar.data.data import Data
from keplar.operator.cleaner import BingoCleaner
from keplar.operator.composite_operator import CompositeOp, CompositeOpReturn
from keplar.operator.creator import BingoCreator, GpCreator
from keplar.operator.crossover import BingoCrossover
from keplar.operator.evaluator import BingoEvaluator
from keplar.operator.generator import BingoGenerator
from keplar.operator.mutation import BingoMutation
from keplar.operator.selector import BingoSelector

# data = Data("txt", "datasets/1.txt",["x","y"])
# data = Data("txt", "datasets/2.txt", ["x0", "x1", "x2", "x3", "x4", "y"])
data = Data("pmlb", "1027_ESL", ["x1", "x2", "x3", 'y'])
fit_list = []
time_list = []
data.read_file()
# data.set_xy("y")
x = data.get_x()
y = data.get_y()
operators = ["+", "-", "*", "/", "^"]
creator = BingoCreator(128, operators, x, 10, "Bingo")
evaluator = BingoEvaluator(x, "exp", "lm", "Bingo", y)
crossover = BingoCrossover("Bingo")
mutation = BingoMutation(x, operators, "Bingo")
selector = BingoSelector(0.5, "tournament", "Bingo")
gen_up_oplist = CompositeOp([crossover, mutation])
gen_down_oplist = CompositeOpReturn([selector])
gen_eva_oplist = CompositeOp([evaluator])
for i in range(10):
    population = creator.do()
    bgsr = KeplarBingoAlg(1000, gen_up_oplist, gen_down_oplist, gen_eva_oplist, 0.001, population)
    bgsr.run()
    # 纯Bingo时可导入bingocpp包
    fit_list.append(bgsr.best_fit)
    time_list.append(bgsr.elapse_time)
fit_pd = pd.DataFrame({'KeplarBingo': fit_list})
time_pd = pd.DataFrame({'KeplarBingo': time_list})
fit_pd.to_csv(r"result/pmlb_1027_result.csv", sep=',', mode="a")
time_pd.to_csv(r"result/pmlb_1027_time_result.csv", sep=',', mode="a")
