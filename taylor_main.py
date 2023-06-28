import numpy as np

from keplar.Algorithm.TaylorGP_Algorithm import TayloGPAlg
from keplar.data.data import Data
from keplar.operator.creator import TaylorGPCreator
from keplar.operator.evaluator import TaylorGPEvaluator
from keplar.operator.selector import TaylorGPSelector
from keplar.operator.crossover import TaylorGPCrossover
from keplar.operator.mutation import TaylorGPMutation
from keplar.operator.reinserter import TaylorGPReinserter
from keplar.operator.Taylor_prework import TaylorGP_Pre1, TaylorGP_pre2

data = Data("pmlb", "1027_ESL", ["x1", "x2", "x3", "x4", 'y'])
data.read_file()
x = data.get_np_x()
y = data.get_np_y()
taylorGP_pre1 = TaylorGP_Pre1(x, y)
X, Y, qualified_list = taylorGP_pre1.do()
print("finish prework!")

taylorGP_pre2 = TaylorGP_pre2(X, Y, qualified_list)
K = taylorGP_pre2.do()
print("finally!")

creator = TaylorGPCreator()
population = creator.do()
evaluator = TaylorGPEvaluator()
eval_op_list = [evaluator]
select = TaylorGPSelector()
crossover = TaylorGPCrossover()
mutation = TaylorGPMutation()
reinsert = TaylorGPReinserter()
taylorGP = TayloGPAlg()
taylorGP.run()

# selector = OperonSelector(5)
# evaluator = OperonEvaluator("R2", x, y, 0.5, True,"Operon")
# crossover = OperonCrossover(x, y, "Operon")
# mutation = OperonMutation(0.9, 0.9, 0.9, 0.5, x, y, 10, 50, "balanced", "Operon")
# reinsert = OperonReinserter(None, "ReplaceWorst", 10, "Operon")
# op_up_list = [mutation, crossover]
# op_down_list = [reinsert]
# eva_list = [evaluator]
# op_alg = OperonAlg(1000, op_up_list, op_down_list, eva_list, selector, 1e-5, 1000, 16, x, y)
# op_alg.run()
