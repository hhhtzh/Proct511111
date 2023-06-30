import numpy as np

from keplar.Algorithm.Alg import OperonBingoAlg
from keplar.data.data import Data
from keplar.operator.creator import OperonCreator
from keplar.operator.crossover import OperonCrossover
from keplar.operator.evaluator import OperonEvaluator
from keplar.operator.mutation import OperonMutation
from keplar.operator.selector import BingoSelector

data = Data("txt", "datasets/1.txt", ["x", "y"])
# data=Data("pmlb","1027_ESL",["x1","x2","x3",'y'])
data.read_file()
data.set_xy("y")
x = data.get_np_x()
y = data.get_np_y()
creator = OperonCreator("balanced", x, y, 100, "Operon")
population = creator.do()
evaluator = OperonEvaluator("R2", x, y, 0.7, True, "self")
eval_op_list = [evaluator]
select = BingoSelector(0.2, "tournament", "Operon")
crossover = OperonCrossover(x, y, "Operon")
mutation = OperonMutation(0.6, 0.7, 0.8, 0.8, x, y, 10, 50, "balanced", "Operon")
op_up_list = [mutation, crossover]
alg = OperonBingoAlg(1000, op_up_list, None, eval_op_list, -10, population, select, x, y, 100)
alg.run()
