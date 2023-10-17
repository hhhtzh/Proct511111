import numpy as np
import pandas as pd

from keplar.Algorithm.Alg import KeplarOperonAlg
from keplar.data.data import Data
from keplar.operator.creator import OperonCreator
from keplar.operator.crossover import OperonCrossover
from keplar.operator.evaluator import OperonEvaluator
from keplar.operator.mutation import OperonMutation
from keplar.operator.selector import OperonSelector

# data = Data("txt", "datasets/1.txt", ["x", "y"])
data = Data("pmlb", "1027_ESL", ["x1", "x2", "x3", 'y'])
# data = Data("txt", "datasets/2.txt", ["x0", "x1", "x2", "x3", "x4", "y"])
# data = Data("pmlb", "529_pollen", ["x1", "x2", "x3", "x4", 'y'])
fit_list = []
time_list = []
data.read_file()
# data.set_xy("y")
x = data.get_np_x()
y = data.get_np_y()
creator = OperonCreator("balanced", x, y, 128, "Operon")
evaluator = OperonEvaluator("RMSE", x, y, 0.7, True, "self")
eval_op_list = [evaluator]
select = OperonSelector(0.2, "tournament", "Operon")
crossover = OperonCrossover(x, y, "Operon")
mutation = OperonMutation(0.6, 0.7, 0.8, 0.8, x, y, 10, 50, "balanced", "Operon")
op_up_list = [mutation, crossover]
for i in range(1):
    population = creator.do()
    alg = KeplarOperonAlg(256, op_up_list, None, eval_op_list, -10, population, select, x, y, 128)
    alg.run()
    fit_list.append(alg.best_fit)
    time_list.append(alg.elapse_time)
# fit_pd = pd.DataFrame({'OperonBingo': fit_list})
# time_pd = pd.DataFrame({'OperonBingo': time_list})
# fit_pd.to_csv(r"result/vla_5.csv", sep=',', mode="a")
# time_pd.to_csv(r"result/vla_5_time.csv", sep=',', mode="a")
