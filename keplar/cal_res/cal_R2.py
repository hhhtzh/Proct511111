

import os

import numpy as np

from sklearn.metrics import r2_score
from sympy import symbols, lambdify
import csv

def calculate_r2(formula,scaler_X,scaler_y, dataset):
    # 读取数据集
    data = np.genfromtxt(dataset, delimiter=',', names=True)
    # data = np.

    # 定义变量
    # variables = [f'x{i+1}' for i in range(len(data.dtype.names)-1)]
    variables = [f'x_{i}' for i in range(len(data.dtype.names)-1)]
    x = symbols(' '.join(variables))
    print("x:", x)

    # 将公式转换为可执行的函数
    formula_func = lambdify(x, formula, dummify=False)
    print("formula_func:", formula_func)
    print("formula", formula)

    # 计算预测值
    X = np.column_stack([data[variable] for variable in data.dtype.names[:-1]])

    X = scaler_X.transform(X)

    # if formula == "0":
    if formula == "0" or formula == "0.0" or formula == 0:

        # 返回固定的零值
        y_pred = np.zeros(len(X))
    else:
        y_pred = formula_func(*X.T)

    # 将矩阵对象转换为数组
    y_true = np.array(data[data.dtype.names[-1]])
    y_true = scaler_y.transform(y_true.reshape(-1, 1))

    # 计算R2值
    r2 = r2_score(y_true, y_pred)
    return r2

def calculate_r2_des(formula, dataset):
    # 读取数据集
    data = np.genfromtxt(dataset, delimiter=',', names=True)

    # 定义变量
    variables = [f'x{i+1}' for i in range(len(data.dtype.names)-1)]
    x = symbols(' '.join(variables))

    # 将公式转换为可执行的函数
    formula_func = lambdify(x, formula, modules=['numpy'])

    # 计算预测值
    X = np.column_stack([data[variable] for variable in data.dtype.names[:-1]])
    # X= 
    y_pred = formula_func(*X.T)

    # 计算R2值
    y_true = data[data.dtype.names[-1]]
    r2 = r2_score(y_true, y_pred)
    return r2

def write_to_csv(filename, formula, r2_score, seeds,complexity):
    # 去除文件名中的路径部分
    filename = filename.replace("./mydatasets/", "")

    # 检查文件是否存在
    file_exists = os.path.isfile("output.csv")

    # 将数据写入 CSV 文件
    with open("output.csv", "a", newline="") as csvfile:
        writer = csv.writer(csvfile)

        # 如果文件不存在或者文件为空，则写入标题行
        if not file_exists or os.stat("output.csv").st_size == 0:
            writer.writerow(["Filename", "Formula", "R2 Score", "Seeds","Complexity"])

        writer.writerow([filename, formula, r2_score, seeds,complexity])