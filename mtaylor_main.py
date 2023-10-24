import numpy as np
import pandas as pd

from keplar.Algorithm.TaylorGP_Algorithm import MTaylorGPAlg
from keplar.data.data import Data
from keplar.population.population import Population
import sys
import argparse
import os
from keplar.cal_res.cal_R2 import calculate_r2
from keplar.cal_res.cal_RMSE import calculate_rmse
# from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

from keplar.utils.utils import Logger

sys.setrecursionlimit(10000)
argparser = argparse.ArgumentParser()
# argparser.add_argument("--trainset", type=str, default="datasets/pmlb/pmlb_txt/523_analcatdata_neavote.txt")
# argparser.add_argument("--varset", type=str, default="datasets/pmlb/pmlb_csv/523_analcatdata_neavote.csv")
argparser.add_argument("--trainset", type=str, default="datasets/feynman/train/feynman-i.12.11.txt")
argparser.add_argument("--varset", type=str, default="datasets/feynman/mydataver/feynman-i.12.11.csv")
args = argparser.parse_args()
print("file path : ", args.trainset)
fileName = os.path.basename(args.trainset)
print("file name : ", fileName)
fileName_whitout_ext = os.path.splitext(fileName)[0]
np.set_printoptions(suppress=True)

pop = Population(128)
fit_list = []
time_list = []
equ_list = []
R2_list = []
rmse_list = []

data = np.loadtxt(args.trainset)
# 将最后一列作为np_y
np_y = data[:, -1]
# 将前面的列作为np_x
np_x = data[:, :-1]

sc_X = StandardScaler()
X_normalized = sc_X.fit_transform(np_x)
sc_y = StandardScaler()
y_normalized = sc_y.fit_transform(np_y.reshape(-1, 1))

# logger = Logger()
# logger.set_log_file("zjw_result/" + fileName_whitout_ext + ".json")
# logger.set_log_level(2)



np_x = np.array(X_normalized)
np_y = np.array(y_normalized)

mt = MTaylorGPAlg(1000, np_x, np_y, population=pop, NewSparseRegressionFlag=True)

for i in range(1):
    # print("iii")
    mt.run()
    # formula = sympify(mt.best_ind)
    r2 = calculate_r2(mt.best_ind, sc_X, sc_y, args.varset)
    rmse = calculate_rmse(mt.best_ind, sc_X, sc_y, args.varset)
    print("-" * 100)
    print("  " * 20 + "R2:", r2)
    print("-" * 100)
    print("  " * 20 + "RMSE:", rmse)
    print("-" * 100)
    rmse_list.append(rmse)
    R2_list.append(r2)
    # fit_list.append(mt.best_fit)
    time_list.append(mt.elapse_time)
    equ_list.append(mt.best_ind)
    print("  " * 20 + "Best fitness:", mt.best_fit)
    print("-" * 100)
    print("  " * 20 + f"Elapse_time: {mt.elapse_time} Seconds")
# fit_pd = pd.DataFrame({'MTaylor(with new sparse)': fit_list})
equ_pd = pd.DataFrame({'MTaylor(with new sparse)': equ_list})
time_pd = pd.DataFrame({'MTaylor(with new sparse)': time_list})
R2_pd = pd.DataFrame({'MTaylor(with new sparse)': R2_list})
rmse_pd = pd.DataFrame({'MTaylor(with new sparse)': rmse_list})

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
rmse_pd.to_csv(r"zjw_result/" + fileName_whitout_ext + "_rmse.csv", sep=',', mode="a")
