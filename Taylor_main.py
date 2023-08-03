import numpy as np

from keplar.Algorithm.TaylorGP_Alg import TayloGPAlg
from keplar.data.data import Data
from keplar.operator.creator import TaylorGPCreator
from keplar.operator.evaluator import TaylorGPEvaluator
from keplar.operator.selector import TaylorGPSelector
from keplar.operator.crossover import TaylorGPCrossover
from keplar.operator.mutation import TaylorGPMutation
from keplar.operator.reinserter import TaylorGPReinserter
from keplar.operator.Taylor_prework import TaylorGP_Pre1, TaylorGP_pre2
from TaylorGP.src.taylorGP.functions import _Function, _sympol_map
from TaylorGP.src.taylorGP.utils import check_random_state
from keplar.operator.TaylorSort import TaylorSort
import pandas as pd

# data = Data("txt", "datasets/1.txt", ["x", "y"])
data = Data("txt", "datasets/2.txt", ["x0", "x1", "x2", "x3", "x4", "y"])

# data = Data("pmlb", "1027_ESL", ["x1", "x2", "x3", "x4", 'y'])
data.read_file()
data.set_xy("y")

x_train = data.get_np_x()
y_train = data.get_np_y()


taylorGP_pre1 = TaylorGP_Pre1(x_train, y_train)
X, Y, qualified_list = taylorGP_pre1.do()

taylorGP_pre2 = TaylorGP_pre2(X, Y, qualified_list)
X, y, params, population_size, seeds, qualified_list, function_set, n_features,sample_wight= taylorGP_pre2.do()

# 生成种群（population）
gen = 0
program = None
creator = TaylorGPCreator(
    X, y, params, gen, population_size, program, "Taylor")
population, sample_wight = creator.do()
print("population_size")
random_state = check_random_state(1)

# 选择最好的一个或者几个
selector = TaylorGPSelector(
    random_state, tournament_size=1000, greater_is_better=False)
# pop_parent,pop_best_index = selector.do(population)
# pop_honor,honor_best_index = selector.do(population)
# pop_now_index=0
pop_parent = 0
pop_honor = 0
pop_now_index = 0
# 做交叉crossover
crossover = TaylorGPCrossover(
    random_state, pop_parent, pop_honor, pop_now_index)


# 做变异，包括子树变异、提升变异（subtree mutation、Hoist mutation、reproduction）
mutation = TaylorGPMutation(1, random_state, pop_parent, pop_now_index)
evaluator = TaylorGPEvaluator(
    "rmse", x_train, y_train, "taylorgp", feature_weight=None)

p_crossover = 0.7,
p_subtree_mutation = 0.01,
p_hoist_mutation = 0.01,
p_point_mutation = 0.3,

method_probs = np.array([p_crossover,
                        p_subtree_mutation,
                        p_hoist_mutation,
                        p_point_mutation])

taylorsort = TaylorSort()

# 算法的全部流程
gen = 100
taylorGP = TayloGPAlg(gen, taylorGP_pre1, taylorGP_pre2, selector,
                      creator, crossover, mutation, method_probs, taylorsort, evaluator)

fit_list = []
time_list = []
for i in range(10):
    taylorGP.run()

    fit_list.append(taylorGP .best_fit)
    time_list.append(taylorGP .elapse_time)
fit_pd = pd.DataFrame({'taylorGP': fit_list})
time_pd = pd.DataFrame({'taylorGP': time_list})
# fit_pd.to_csv(r"result/pmlb_1027_result.csv", sep=',', mode="a")
# time_pd.to_csv(r"result/pmlb_1027_time_result.csv", sep=',', mode="a")
fit_pd.to_csv(r"result/vla_5.csv", sep=',', mode="a")
time_pd.to_csv(r"result/vla_5_time.csv", sep=',', mode="a")

