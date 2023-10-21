

import os

import numpy as np

from sklearn.metrics import r2_score
from sympy import symbols, lambdify
import csv

from keplar.data.data import Data
from keplar.operator.evaluator import OperonSingleEvaluator
def is_float(value):
    try:
        float(value)
        return True
    except ValueError:
        return False
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
    elif is_float(formula):
        # 返回固定的常数值
        print("len X:",len(X))
        y_pred = np.full(len(X), float(formula))
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


if __name__ == '__main__':

    # formula ='-0.00200000000000000'
    # formula ="-0.00200000000000000"
    formula =-0.00200000000000000
    # formula = 0
    # formula ="-0.002000"
    print("formula:", formula)
    # formula = "-0.002000+167x"
    # formula = sympify(formula)
    sys.setrecursionlimit(10000)
    argparser = argparse.ArgumentParser()
    # argparser.add_argument("--trainset", type=str, default="../../datasets/pmlb/pmlb_txt/523_analcatdata_neavote.txt")
    # argparser.add_argument("--varset", type=str, default="../../datasets/pmlb/pmlb_csv/523_analcatdata_neavote.csv")
    argparser.add_argument("--trainset", type=str, default="../../datasets/feynman/train/feynman-i.12.11.txt")
    argparser.add_argument("--varset", type=str, default="../../datasets/feynman/mydataver/feynman-i.12.11.csv")
    args = argparser.parse_args()
    print("file path : ", args.trainset)
    fileName = os.path.basename(args.trainset)
    print("file name : ", fileName)
    fileName_whitout_ext = os.path.splitext(fileName)[0]
    # np.set_printoptions(suppress=True)

    # pop = Population(128)
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

    # r2 = calculate_r2(formula, sc_X, sc_y, args.varset)
    rmse = calculate_rmse(formula, sc_X, sc_y, args.varset)
    # print("r2:", r2)
    print("rmse:", rmse)
