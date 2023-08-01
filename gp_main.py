from time import time

import pandas as pd

from gplearn.genetic import SymbolicRegressor
from keplar.data.data import Data
from keplar.operator.evaluator import SingleBingoEvaluator
from keplar.translator.translator import gp_to_bingo, gp_to_bingo1

# data = Data("pmlb", "1027_ESL", ["x1", "x2", "x3", "x4",'y'])
# data.read_file()
data = Data("txt", "datasets/2.txt", ["x0", "x1", "x2", "x3", "x4", "y"])
data.read_file()
data.set_xy("y")
fit_list = []
time_list = []
x = data.get_np_x()
y = data.get_np_y()
sr = SymbolicRegressor(population_size=128, generations=1000,
                       function_set=["add", "sub", "mul", "div", "sin", "cos", 'power'])
for i in range(10):
    t1=time()
    tt = sr.fit(X=x, y=y)
    aa = gp_to_bingo1(str(tt))
    print(aa)
    eval = SingleBingoEvaluator(data, equation=str(aa))
    fit=eval.do()
    fit_list.append(fit)
    et=time()-t1
    time_list.append(et)
fit_pd = pd.DataFrame({'Gplearn': fit_list})
time_pd = pd.DataFrame({'Gplearn': time_list})
fit_pd.to_csv(r"result/vla_5.csv", sep=',', mode="a")
time_pd.to_csv(r"result/vla_5_time.csv", sep=',', mode="a")
# print(str(tt))
# aa = gp_to_bingo1(str(tt))
# eval = SingleBingoEvaluator(data, equation=str(aa))
# print(eval.do())
