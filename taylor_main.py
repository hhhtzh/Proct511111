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
from TaylorGP.src.taylorGP.functions import _Function,_sympol_map
from TaylorGP.src.taylorGP.utils import check_random_state


data = Data("pmlb", "1027_ESL", ["x1", "x2", "x3", "x4", 'y'])
data.read_file()
x_train = data.get_np_x()
y_train = data.get_np_y()
taylorGP_pre1 = TaylorGP_Pre1(x_train, y_train)
X, Y, qualified_list = taylorGP_pre1.do()
# print("finish prework!")

taylorGP_pre2 = TaylorGP_pre2(X, Y, qualified_list)
X,y,params,population_size,seeds,qualified_list,function_set,n_features= taylorGP_pre2.do()
# print("finally!")
# print(population_size)

# print(K[110].__str__())
# print(str(K[110]))

# for i, node in enumerate(program[110].program):
#             if isinstance(node, _Function):
#                     print(node.name)
#             else:
#                     print(node)

#生成种群（population）
gen =0
program = None
creator = TaylorGPCreator(X,y,params,gen,population_size,program,"Taylor")
population= creator.do()

print("population_size")

#计算fitness的值
# evaluator = TaylorGPEvaluator() 
# eval_op_list = [evaluator]
random_state = check_random_state(1)

#选择最好的一个或者几个
# tournament_size = 1000
# greater_is_better = False
selector = TaylorGPSelector(random_state,tournament_size=1000,greater_is_better=False)
pop_parent,pop_best_index = selector.do(population)
pop_honor,honor_best_index = selector.do(population)
pop_now_index=0

print("crossover begin!")


#做交叉crossover
crossover = TaylorGPCrossover(random_state,qualified_list,function_set,n_features,pop_parent.program,pop_honor.program,pop_now_index)
# population= crossover.do(population)
print("crossover end!")



#做变异，包括子树变异、提升变异（subtree mutation、Hoist mutation、reproduction）
# option = 1
mutation = TaylorGPMutation(1,random_state,qualified_list,function_set,n_features,pop_parent,pop_now_index)
# mutation2 = TaylorGPMutation(2,random_state,qualified_list,function_set,n_features,pragram_useless,pop_parent,pop_best_index)
# mutation3 = TaylorGPMutation(3,random_state,qualified_list,function_set,n_features,pragram_useless,pop_parent,pop_best_index)
# mutation4 = TaylorGPMutation(4,random_state,qualified_list,function_set,n_features,pragram_useless,pop_parent,pop_best_index)
evaluator = TaylorGPEvaluator("rmse",x_train,y_train,"taylorgp",feature_weight=None)

# population = mutation1.do(population)

p_crossover=0.9,
p_subtree_mutation=0.01,
p_hoist_mutation=0.01,
p_point_mutation=0.01,

method_probs = np.array([p_crossover,
                        p_subtree_mutation,
                        p_hoist_mutation,
                        p_point_mutation])

# mutation =[mutation1,mutation2,mutation3,mutation4]


#算法的全部流程
gen = 1
taylorGP = TayloGPAlg(gen,taylorGP_pre1,taylorGP_pre2,selector,creator,crossover,mutation,method_probs,evaluator)

taylorGP.run()

