import pandas as pd

from keplar.Algorithm.Alg import BingoAlg
from keplar.data.data import Data

data = Data("pmlb", "1027_ESL", ["x1", "x2", "x3", 'y'])
# data = Data("txt", "datasets/2.txt", ["x0", "x1", "x2", "x3", "x4", "y"])
data.read_file()
# data.set_xy("y")
fit_list = []
time_list = []
bingo = BingoAlg(1000, data,
                 operators=["+", "-", "*", "/", "sin", "exp", "cos", 'sqrt', 'log', 'sin', 'pow', 'exp', '^'])
for i in range(10):
    bingo.run()
    fit_list.append(bingo.island.get_best_fitness())
    time_list.append(bingo.elapse_time)

fit_pd = pd.DataFrame({'BingoCPP': fit_list})
time_pd = pd.DataFrame({'BingoCPP': time_list})
fit_pd.to_csv(r"result/pmlb_1027_result.csv", sep=',')
time_pd.to_csv(r"result/pmlb_1027_time_result.csv", sep=',')
