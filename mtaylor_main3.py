import numpy as np
import pandas as pd

from keplar.Algorithm.TaylorGP_Algorithm import MTaylorGPAlg
from keplar.data.data import Data
from keplar.population.population import Population

# data = Data("txt_pmlb", "datasets/pmlb/val/197_cpu_act.txt", ["x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12", "x13", "x14", "x15", "x16", "x17", "x18", "x19", "x20","y"])
# data = Data("txt_pmlb", "datasets/pmlb/val/503_wind.txt", ["x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12", "x13","y"])
data = Data("txt_pmlb", "datasets/feynman/train/feynman-i.12.1.txt", ["x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "y"])
# data = Data("txt", "datasets/vla/two/1.txt", ["x0", "x1", "y"])
# data = Data("pmlb", "1027_ESL", ["x1", "x2", "x3", "x4", 'y'])
# data = Data("txt_pmlb", "datasets/pmlb/train/529_pollen.txt", ["x0", "x1", "x2", "x3", "y"])
np.set_printoptions(suppress=True)
data.read_file()
data.set_xy("y")
pop = Population(128)
fit_list = []
time_list = []
equ_list=[]
mt = MTaylorGPAlg(1000, data, population=pop, NewSparseRegressionFlag=True)
for i in range(10):
    mt.run()
# fit_list.append(mt.best_fit)
    time_list.append(mt.elapse_time)
    equ_list.append(mt.best_ind)
# print(mt.best_fit)# print(mt.elapse_time)
# fit_pd = pd.DataFrame({'MTaylor(with new sparse)': fit_list})
equ_pd=pd.DataFrame({'MTaylor(with new sparse)': equ_list})
time_pd = pd.DataFrame({'MTaylor(with new sparse)': time_list})
# fit_pd = pd.DataFrame({'MTaylor': fit_list})
# time_pd = pd.DataFrame({'MTaylor': time_list})
# fit_pd.to_csv(r"result/vla_5.csv", sep=',', mode="a")
# time_pd.to_csv(r"result/vla_5_time.csv", sep=',', mode="a")
# fit_pd.to_csv(r"result/197_cpu_act.csv", sep=',', mode="a")
equ_pd.to_csv(r"result/pmlb_608_fri_c3_1000_10_equ.csv", sep=',', mode="a")
time_pd.to_csv(r"result/pmlb_608_fri_c3_1000_10_time.csv", sep=',', mode="a")
# fit_pd.to_csv(r"result/pmlb_529_pollen.csv", sep=',', mode="a")
# time_pd.to_csv(r"result/pmlb_529_pollen_time.csv", sep=',', mode="a")
# equ_pd.to_csv(r"result/pmlb_529_pollen_equ.csv", sep=',', mode="a")