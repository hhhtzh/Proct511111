

import os

import numpy as np

from sklearn.metrics import r2_score
from sympy import symbols, lambdify
import csv

from keplar.data.data import Data
from keplar.operator.evaluator import OperonSingleEvaluator

def calculate_rmse_cal(y_pred, y_true):
    # 将 y_pred 和 y_true 转换为 numpy 数组
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    z= y_pred - y_true

    # 计算均方根误差
    mse = np.mean((y_pred - y_true) ** 2)
    rmse = np.sqrt(mse)

    return rmse


def calculate_rmse(formula,scaler_X,scaler_y, dataset):
    # 读取数据集
    data = np.genfromtxt(dataset, delimiter=',', names=True)
    # data = np.

    # 定义变量
    # variables = [f'x{i+1}' for i in range(len(data.dtype.names)-1)]
    variables = [f'x_{i}' for i in range(len(data.dtype.names)-1)]
    x = symbols(' '.join(variables))

    # 将公式转换为可执行的函数
    formula_func = lambdify(x, formula, dummify=False)
    # print("formula_func:", formula_func)
    # print("formula", formula)

    # 计算预测值
    X = np.column_stack([data[variable] for variable in data.dtype.names[:-1]])
    # print("X:", X)

    X = scaler_X.transform(X)
    # print("X:", X)

    # if formula == "0":
    if formula == "0" or formula == "0.0" or formula == 0:

        # 返回固定的零值
        y_pred = np.zeros(len(X))
    else:
        y_pred = formula_func(*X.T)
    
    y_pred = y_pred.reshape(-1, 1)

    # 将矩阵对象转换为数组
    y_true = np.array(data[data.dtype.names[-1]])
    y_true = scaler_y.transform(y_true.reshape(-1, 1))
    # print("y_true:", y_true)
    # print("y_pred:", y_pred)

    fit = calculate_rmse_cal(y_pred, y_true)

    # eval = OperonSingleEvaluator("RMSE", X, y_true, 1, True, op_equ=formula)
    # fit = eval.do()
    return fit

