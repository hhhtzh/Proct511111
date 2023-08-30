import numpy as np
import pandas as pd

from keplar.Algorithm.operon_Algorithm import OperonPlus, KeplarMOperon
from keplar.data.data import Data
from keplar.population.population import Population

fit_list = []
time_list = []
# data = Data("pmlb", "1027_ESL", ["x1", "x2", "x3", "x4", 'y'])
data = Data("txt", "datasets/2.txt", ["x0", "x1", "x2", "x3", "x4", "y"])
data.read_file()
data.set_xy("y")
data.read_file()
print(np.shape(data.get_np_ds()))
operators = ["+", "-", "*", "/", "sin", "exp", "sqrt", "^"]
population = Population(128)
kmo = KeplarMOperon(1000, None, None, None, 0.1, population, data, operators,p_noise=0)
for i in range(10):
    kmo.run()
    fit_list.append(kmo.best_fit)
    time_list.append(kmo.elapse_time)
fit_pd = pd.DataFrame({'KeplarMOperon': fit_list})
time_pd = pd.DataFrame({'KeplarMOperon': time_list})
fit_pd.to_csv(r"result/vla_5.csv", sep=',', mode="a")
time_pd.to_csv(r"result/vla_5_time.csv", sep=',', mode="a")