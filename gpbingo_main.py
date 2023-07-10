from pmlb import fetch_data
from keplar.Algorithm.Alg import KeplarBingoAlg, GpBingoAlg

from keplar.data.data import Data
from keplar.operator.cleaner import BingoCleaner
from keplar.operator.composite_operator import CompositeOp, CompositeOpReturn
from keplar.operator.creator import BingoCreator, GpCreator
from keplar.operator.crossover import BingoCrossover
from keplar.operator.evaluator import BingoEvaluator
from keplar.operator.generator import BingoGenerator
from keplar.operator.mutation import BingoMutation
from keplar.operator.selector import BingoSelector

# data = Data("txt", "datasets/1.txt", ["x", "y"])
data = Data("pmlb", "1027_ESL", ["x1", "x2", "x3", 'y'])
data.read_file()
# data.set_xy("y")
x = data.get_np_x()
y = data.get_np_y()
operators =["+", "-", "*", "/", "sin", "exp", "cos", 'sqrt', 'log', 'sin', 'pow', 'exp', '^']
creator = GpCreator(128, x, y, "Bingo", n_jobs=20)
evaluator = BingoEvaluator(x, "exp", "lm", "Bingo", y, metric="rmse")
crossover = BingoCrossover("Bingo")
mutation = BingoMutation(x, operators, "Bingo")
selector = BingoSelector(0.5, "tournament", "Bingo")
gen_up_oplist = CompositeOp([crossover, mutation])
gen_down_oplist = CompositeOpReturn([selector])
gen_eva_oplist = CompositeOp([evaluator])
population = creator.do()
bgsr = GpBingoAlg(1000, gen_up_oplist, gen_down_oplist, gen_eva_oplist, 0.001, population)
bgsr.run()
