import pandas as pd

from keplar.Algorithm.Alg import BingoAlg
from keplar.data.data import Data
from keplar.operator.statistic import BingoStatistic

# data = Data("pmlb", "1027_ESL", ["x1", "x2", "x3", 'y'])
data = Data("txt", "datasets/2.txt", ["x0", "x1", "x2", "x3", "x4", "y"])
data.read_file()
data.set_xy("y")
fit_list = []
time_list = []
bingo = BingoAlg(1000, data,
                 operators=["+", "-", "*", "/", "sin", "exp", "cos", 'sqrt', 'log', 'sin', 'pow', 'exp', '^'])
for i in range(10):
    bingo.run()
    fit_list.append(bingo.best_fit)
    time_list.append(bingo.elapse_time)

fit_pd = pd.DataFrame({'Bingo': fit_list})
time_pd = pd.DataFrame({'Bingo': time_list})
fit_pd.to_csv(r"result/vla_5.csv", sep=',', mode="a")
time_pd.to_csv(r"result/vla_5_time.csv", sep=',', mode="a")
# str2 = str(bingo.best_ind)
# sta = BingoStatistic(str2)
# sta.pos_do()
