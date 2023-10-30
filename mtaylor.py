import numpy as np
import pandas as pd

from keplar.Algorithm.TaylorGP_Algorithm import MTaylorGPAlg
from keplar.data.data import Data
from keplar.population.population import Population
import sys
import argparse
import os
from keplar.cal_res.cal_R2 import calculate_r2,cal_r2
from keplar.cal_res.cal_RMSE import calculate_rmse,cal_rmse
# from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

from keplar.utils.utils import Logger
from pmlb.pmlb import fetch_data
from sklearn.model_selection import train_test_split

def print_stand(r2,rmse,fitness,elapse_time):
    print("-" * 100)
    print("  " * 20 + "R2:", r2)
    print("-" * 100)
    print("  " * 20 + "RMSE:", rmse)
    print("-" * 100)
    print("  " * 20 + "Best fitness:", fitness)
    print("-" * 100)
    print("  " * 20 + f"Elapse_time: {elapse_time} Seconds")


if __name__ == '__main__':
    sys.setrecursionlimit(10000)
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--name", type=str, default="feynman_I_12_5")
    args = argparser.parse_args()
    print("file name : ", args.name)

    X,y = fetch_data(args.name, return_X_y=True,local_cache_dir="/home/friday/Documents/pmlb/datasets/")
    print("read data success!")
    # X,y = fetch_data(args.name, return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    print("X_train shape:", X_train.shape)

    pop = Population(128)
    fit_list = []
    time_list = []
    equ_list = []
    R2_list = []
    rmse_list = []

    if "feynman" in args.name:
        scaler_X = False
    else:
        scaler_X = True

    if scaler_X:
        sc_X = StandardScaler()
        X_normalized = sc_X.fit_transform(X_train)
        sc_y = StandardScaler()
        y_normalized = sc_y.fit_transform(y_train.reshape(-1, 1))

        np_x = np.array(X_normalized)
        np_y = np.array(y_normalized)
    else:
        sc_X = None
        sc_y = None
        np_x = np.array(X_train)
        np_y = np.array(y_train)

    mt = MTaylorGPAlg(1000, np_x, np_y, population=pop, NewSparseRegressionFlag=True)

    for i in range(1):
        mt.run()
        # forual = "0.31"
        # r2 = cal_r2(forual, sc_X, sc_y, X_test, y_test,scaler_X)
        # rmse = cal_rmse(forual, sc_X, sc_y, X_test, y_test,scaler_X)
        # print("r2:",r2)
        # print("rmse:",rmse)
        # print_stand(r2, rmse, forual, 2)


        r2 = cal_r2(mt.best_ind, sc_X, sc_y, X_test, y_test,scaler_X)
        rmse = cal_rmse(mt.best_ind, sc_X, sc_y, X_test, y_test,scaler_X)

        print_stand(r2, rmse, mt.best_ind, mt.elapse_time)


        rmse_list.append(rmse)
        R2_list.append(r2)
        time_list.append(mt.elapse_time)
        equ_list.append(mt.best_ind)
        equ_pd = pd.DataFrame({'MTaylor(with new sparse)': equ_list})
        time_pd = pd.DataFrame({'MTaylor(with new sparse)': time_list})
        R2_pd = pd.DataFrame({'MTaylor(with new sparse)': R2_list})
        rmse_pd = pd.DataFrame({'MTaylor(with new sparse)': rmse_list})
        time_pd.to_csv(r"res/" + args.name + "_time.csv", sep=',', mode="a")
        equ_pd.to_csv(r"res/" + args.name + "_equ.csv", sep=',', mode="a")
        R2_pd.to_csv(r"res/" + args.name + "_R2.csv", sep=',', mode="a")
        rmse_pd.to_csv(r"res/" + args.name + "_rmse.csv", sep=',', mode="a")
