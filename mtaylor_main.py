import pandas as pd

from keplar.Algorithm.TaylorGP_Algorithm import MTaylorGPAlg
from keplar.data.data import Data
from keplar.population.population import Population

data = Data("txt", "datasets/2.txt", ["x0", "x1", "x2", "x3", "x4", "y"])
# data = Data("pmlb", "1027_ESL", ["x1", "x2", "x3", 'y'])
data.read_file()
data.set_xy("y")
pop = Population(100)
fit_list = []
time_list = []
mt = MTaylorGPAlg(1000, data, population=pop,SR_method="Bingo")
for i in range(10):
    mt.run()
    fit_list.append(mt.best_fit)
    time_list.append(mt.elapse_time)

fit_pd = pd.DataFrame({'MTaylor': fit_list})
time_pd = pd.DataFrame({'MTaylor': time_list})
fit_pd.to_csv(r"result/pmlb_1027_result.csv", sep=',', mode="a")
time_pd.to_csv(r"result/pmlb_1027_time_result.csv", sep=',', mode="a")
