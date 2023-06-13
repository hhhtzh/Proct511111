import numpy as np

from keplar.Algorithm.operon_Algorithm import OperonAlg
from keplar.data.data import Data
from keplar.operator.creator import OperonCreator
from keplar.operator.crossover import OperonCrossover
from keplar.operator.evaluator import OperonEvaluator
from keplar.operator.mutation import OperonMutation
from keplar.operator.reinserter import OperonReinserter
from keplar.operator.selector import OperonSelector

data = Data("pmlb", '1027_ESL')
data.read_file()
x = data.get_x()
y = data.get_y()
x = np.array(x)

y = np.array(y)
selector = OperonSelector(5)
evaluator = OperonEvaluator("R2", x, y, 0.5, True)
crossover = OperonCrossover(x, y, "Operon")
mutation = OperonMutation(0.9, 0.9, 0.9, 0.5, x, y, 10, 50, "balanced", "Operon")
reinsert = OperonReinserter(None, "ReplaceWorst", 10, "Operon")
op_up_list = [mutation, crossover]
op_down_list = [reinsert]
eva_list = [evaluator]
op_alg = OperonAlg(1000, op_up_list, op_down_list, eva_list, selector, 1e-5, 1000, 16, x, y)
op_alg.run()
