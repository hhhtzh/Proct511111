import pandas as pd

from keplar.Algorithm.operon_Algorithm import OperonPlus
from keplar.data.data import Data
from keplar.population.population import Population

fit_list = []
time_list = []
data = Data("pmlb", "1027_ESL", ["x1", "x2", "x3", 'y'])
data.read_file()
operators = ["+", "-", "*", "/", "sin", "exp", "sqrt", "^"]
population = Population(128)
kmo = OperonPlus(1000, None, None, None, 0.1, population, data, operators)
for i in range(10):
    kmo.run()
    fit_list.append(kmo.best_fit)
    time_list.append(kmo.elapse_time)
fit_pd = pd.DataFrame({'KeplarOperon': fit_list})
time_pd = pd.DataFrame({'KeplarOperon': time_list})
fit_pd.to_csv(r"result/pmlb_1027_result.csv", sep=',', mode="a")
time_pd.to_csv(r"result/pmlb_1027_time_result.csv", sep=',', mode="a")