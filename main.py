import numpy as np
from pmlb import fetch_data

from keplar.Algorithm.sralg import BingoSRAlg
from keplar.data.data import Data
from keplar.operator.cleaner import BingoCleaner
from keplar.operator.composite_operator import CompositeOp, CompositeOpReturn
from keplar.operator.creator import BingoCreator, GpCreator, OperonCreator
from keplar.operator.crossover import BingoCrossover
from keplar.operator.evaluator import BingoEvaluator
from keplar.operator.generator import BingoGenerator
from keplar.operator.mutation import BingoMutation
from keplar.operator.selector import BingoSelector


dt = Data("pmlb", "1027_ESL")
dt.read_file()
x=dt.get_x()
y=dt.get_y()
x=np.array(x)
operators = ["+", "-", "*", "/","exp","^","sin"]
creator = OperonCreator("balanced", x, y, 128)
# creator=BingoCreator(128,operators,x,10)
evaluator = BingoEvaluator(x, "exp", "TNC", y)
crossover = BingoCrossover()
mutation = BingoMutation(x, operators)
selector = BingoSelector(0.5, "tournament")
gen_up_oplist = CompositeOp([crossover, mutation])
gen_down_oplist = CompositeOpReturn([selector])
gen_eva_oplist = CompositeOp([evaluator])
population = creator.do()

bgsr = BingoSRAlg(100, gen_up_oplist, gen_down_oplist, gen_eva_oplist, 0.001, population)
bgsr.run()
