from pmlb import fetch_data

from keplar.Algorithm.sralg import BingoSRAlg
from keplar.operator.cleaner import BingoCleaner
from keplar.operator.composite_operator import CompositeOp, CompositeOpReturn
from keplar.operator.creator import BingoCreator, GpCreator
from keplar.operator.crossover import BingoCrossover
from keplar.operator.evaluator import BingoEvaluator
from keplar.operator.generator import BingoGenerator
from keplar.operator.mutation import BingoMutation
from keplar.operator.selector import BingoSelector

x, y = fetch_data('1027_ESL', return_X_y=True, local_cache_dir='./datasets')
operators = ["+", "-", "*", "/"]
creator = GpCreator(50, x, y)
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
