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

# for i, node in enumerate(program[110].program):
#             if isinstance(node, _Function):
#                     print(node.name)
#             else:
#                     print(node)

#生成种群（population）
creator = TaylorGPCreator(program,"Taylor")
population = creator.do()


#计算fitness的值
evaluator = TaylorGPEvaluator()
eval_op_list = [evaluator]

#选择最好的一个或者几个
select = TaylorGPSelector()

#做交叉crossover
crossover = TaylorGPCrossover()

#做变异，包括子树变异、提升变异（subtree mutation、Hoist mutation、reproduction）
mutation = TaylorGPMutation()

#算法的全部流程
taylorGP = TayloGPAlg()

taylorGP.run()

