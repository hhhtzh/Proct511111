import numpy as np
import pandas as pd

from keplar.Algorithm.TaylorGP_Algorithm import MTaylorGPAlg
from keplar.data.data import Data
from keplar.population.population import Population
import sys
import argparse
import os
from keplar.cal_res.cal_R2 import calculate_r2
# data = Data("txt", "trainsets/pmlb/val/197_cpu_act.txt", ["x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12", "x13", "x14", "x15", "x16", "x17", "x18", "x19", "x20","y"])
# data = Data("txt", "trainsets/vla/two/1.txt", ["x0", "x1", "y"])
sys.setrecursionlimit(10000)
argparser = argparse.ArgumentParser()
argparser.add_argument("--trainset", type=str, default="datasets/pmlb/train/588_fri_c4_1000_100.txt")
argparser.add_argument("--varset", type=str, default="datasets/pmlb/val/588_fri_c4_1000_100.csv")
args = argparser.parse_args()
print("file path : ", args.trainset)
fileName = os.path.basename(args.trainset)
print("file name : ", fileName)
fileName_whitout_ext = os.path.splitext(fileName)[0]

# data = Data("txt_pmlb", args.trainset, ["x0", "x1","x2","x3","x4", "y"])
# data=Data("numpy",args.trainset)

# data2 = Data("txt_pmlb", args.varset, ["x0", "x1", "y"])

np.set_printoptions(suppress=True)
# data.read_file()
# data.set_xy("y")
pop = Population(128)
fit_list = []
time_list = []
equ_list = []
R2_list = []
mt =  MTaylorGPAlg(1000, args.trainset, population=pop, NewSparseRegressionFlag=True)





for i in range(1):
    print("iii")
    mt.run()

    r2=calculate_r2(mt.best_ind, args.varset)
    print("r2:",r2)
    R2_list.append(r2)
    # fit_list.append(mt.best_fit)
    time_list.append(mt.elapse_time)
    equ_list.append(mt.best_ind)
    print(mt.best_fit)
    print(mt.elapse_time)
# fit_pd = pd.DataFrame({'MTaylor(with new sparse)': fit_list})
equ_pd = pd.DataFrame({'MTaylor(with new sparse)': equ_list})
time_pd = pd.DataFrame({'MTaylor(with new sparse)': time_list})
R2_pd = pd.DataFrame({'MTaylor(with new sparse)': R2_list})
# fit_pd = pd.DataFrame({'MTaylor': fit_list})
# time_pd = pd.DataFrame({'MTaylor': time_list})
# fit_pd.to_csv(r"result/vla_5.csv", sep=',', mode="a")
# time_pd.to_csv(r"result/vla_5_time.csv", sep=',', mode="a")
# fit_pd.to_csv(r"result/197_cpu_act.csv", sep=',', mode="a")
# time_pd.to_csv(r"result/197_cpu_act_time.csv", sep=',', mode="a")
# fit_pd.to_csv(r"result/pmlb_529_pollen.csv", sep=',', mode="a")
time_pd.to_csv(r"zjw_result/" + fileName_whitout_ext + "_time.csv", sep=',', mode="a")
equ_pd.to_csv(r"zjw_result/" + fileName_whitout_ext + "_equ.csv", sep=',', mode="a")
R2_pd.to_csv(r"zjw_result/" + fileName_whitout_ext + "_R2.csv", sep=',', mode="a")
