import numpy as np
import pandas as pd

from keplar.Algorithm.operon_Algorithm import OperonAlg
from keplar.data.data import Data
from keplar.operator.creator import OperonCreator
from keplar.operator.crossover import OperonCrossover
from keplar.operator.evaluator import OperonEvaluator
from keplar.operator.mutation import OperonMutation
from keplar.operator.reinserter import OperonReinserter
from keplar.operator.selector import OperonSelector

# data = Data("txt", "datasets/1.txt", ["x", "y"])
# data = Data("txt", "datasets/2.txt", ["x0", "x1","x2","x3","x4","y"])

data = Data("pmlb", "1027_ESL", ["x1", "x2", "x3", 'y'])
data.read_file()
# data.set_xy("y")
fit_list = []
time_list = []
x = data.get_np_x()
y = data.get_np_y()
selector = OperonSelector(5)
evaluator = OperonEvaluator("RMSE", x, y, 0.5, True, "Operon")
crossover = OperonCrossover(x, y, "Operon")
mutation = OperonMutation(1, 1, 1, 0.5, x, y, 10, 50, "balanced", "Operon")
reinsert = OperonReinserter(None, "ReplaceWorst", 10, "Operon", x, y)
op_up_list = [mutation, crossover]
op_down_list = [reinsert]
eva_list = [evaluator]
op_alg = OperonAlg(10000, op_up_list, op_down_list, eva_list, selector, 1e-5, 1000, 16, x, y)
# for i in range(10):
op_alg.run()
# print(op_alg.model_string)
# op_alg.get_n_top()
#     fit_list.append(op_alg .best_fit)
#     time_list.append(op_alg .elapse_time)
# fit_pd = pd.DataFrame({'Operon': fit_list})
# time_pd = pd.DataFrame({'Operon': time_list})
# fit_pd.to_csv(r"result/pmlb_1027_result.csv", sep=',', mode="a")
# time_pd.to_csv(r"result/pmlb_1027_time_result.csv", sep=',', mode="a")
