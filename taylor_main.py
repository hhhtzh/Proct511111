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
from  TaylorGP.src.taylorGP.functions import _Function,_sympol_map


data = Data("pmlb", "1027_ESL", ["x1", "x2", "x3", "x4", 'y'])
data.read_file()
x = data.get_np_x()
y = data.get_np_y()
taylorGP_pre1 = TaylorGP_Pre1(x, y)
X, Y, qualified_list = taylorGP_pre1.do()
print("finish prework!")

taylorGP_pre2 = TaylorGP_pre2(X, Y, qualified_list)
program = taylorGP_pre2.do()
print("finally!")

# print(K[110].__str__())
# print(str(K[110]))

# for i, node in enumerate(K[110].program):
#             if isinstance(node, _Function):
#                     print(node.name)
#             else:
#                     print(node)

creator = TaylorGPCreator(program)
population = creator.do()

# for i,node in K[110]:
#     if isinstance(node, _Function):
#         print(node.name)
# print(K[110].__str__())


# creator = TaylorGPCreator()
# population = creator.do()

evaluator = TaylorGPEvaluator()
eval_op_list = [evaluator]
select = TaylorGPSelector()

crossover = TaylorGPCrossover()
mutation = TaylorGPMutation()
# reinsert = TaylorGPReinserter()
taylorGP = TayloGPAlg()
# taylorGP = TayloGPAlg(20,sdakdj,sd,)
taylorGP.run()

