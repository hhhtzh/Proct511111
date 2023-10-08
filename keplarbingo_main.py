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

# json[operator][operator_list]=
# data = Data("txt", "datasets/1.txt",["x","y"])
# data = Data("txt", "datasets/2.txt", ["x0", "x1", "x2", "x3", "x4", "y"])
# data = Data("txt", "datasets/vla/two/1.txt", ["x0", "x1", "y"])
data = Data("pmlb", "1027_ESL", ["x1", "x2", "x3", 'y'])
fit_list = []
time_list = []
data.read_file()
# data.set_xy("y")
x = data.get_x()
y = data.get_y()
operators = ["+", "-", "*", "/", "sin", "exp", "cos", 'sqrt', 'log', 'sin', 'pow', 'exp', '^']
creator = BingoCreator(128, operators, x, 10, "Bingo")
evaluator = BingoEvaluator(x, "exp", "lm", "Bingo", y)
crossover = BingoCrossover("Bingo")
mutation = BingoMutation(x, operators, "Bingo")
selector = BingoSelector(0.5, "tournament", "Bingo")
gen_up_oplist = CompositeOp([crossover, mutation])
gen_down_oplist = CompositeOpReturn([selector])
gen_eva_oplist = CompositeOp([evaluator])
for i in range(1):
    population = creator.do()
    bgsr = KeplarBingoAlg(10000, gen_up_oplist, gen_down_oplist, gen_eva_oplist, 0.001, population)
    bgsr.run()
    # 纯Bingo时可导入bingocpp包
    fit_list.append(bgsr.best_fit)
    time_list.append(bgsr.elapse_time)
# fit_pd = pd.DataFrame({'KeplarBingoCPP': fit_list})
# time_pd = pd.DataFrame({'KeplarBingoCPP': time_list})
# fit_pd.to_csv(r"result/vla_5.csv", sep=',', mode="a")
# time_pd.to_csv(r"result/vla_5_time.csv", sep=',', mode="a")
