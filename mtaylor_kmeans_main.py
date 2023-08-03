import pandas as pd

from keplar.Algorithm.TaylorGP_Algorithm import MTaylorKMeansAlg
from keplar.data.data import Data
from keplar.population.population import Population


# data = Data("txt", "datasets/vla/two/5.txt", ["x0", "x1", "y"])
data = Data("txt", "datasets/2.txt", ["x0", "x1", "x2", "x3", "x4", "y"])
# data = Data("pmlb", "1027_ESL", ["x1", "x2", "x3", "x4", 'y'])
data.read_file()
data.set_xy("y")
pop = Population(128)
fit_list = []
time_list = []
mt = MTaylorKMeansAlg(1000, data, population=pop, repeat=10)
for i in range(10):
    mt.run()

    fit_list.append(mt.best_fit)
    time_list.append(mt.elapse_time)

fit_pd = pd.DataFrame({'MTaylorKMeans': fit_list})
time_pd = pd.DataFrame({'MTaylorKMeans': time_list})
fit_pd.to_csv(r"result/vla_5.csv", sep=',', mode="a")
time_pd.to_csv(r"result/vla_5_time.csv", sep=',', mode="a")
