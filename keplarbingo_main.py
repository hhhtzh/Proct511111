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
data = Data("txt", "datasets/2.txt", ["x0", "x1", "x2", "x3", "x4", "y"])
data.read_file()
data.set_xy("y")
x = data.get_x()
y = data.get_y()
operators = ["+", "-", "*", "/", "^"]
creator = BingoCreator(50, operators, x, 10, "Bingo")
evaluator = BingoEvaluator(x, "exp", "lm", "Bingo", y)
crossover = BingoCrossover("Bingo")
mutation = BingoMutation(x, operators, "Bingo")
selector = BingoSelector(0.5, "tournament", "Bingo")
gen_up_oplist = CompositeOp([crossover, mutation])
gen_down_oplist = CompositeOpReturn([selector])
gen_eva_oplist = CompositeOp([evaluator])
population = creator.do()
bgsr = KeplarBingoAlg(1000, gen_up_oplist, gen_down_oplist, gen_eva_oplist, 0.001, population)
bgsr.run()
# 纯Bingo时可导入bingocpp包
