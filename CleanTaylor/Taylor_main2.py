import sys
import argparse
import numpy as np
from sympy import *
from CalTaylor import Metrics
from _global import _init, set_value, get_value
import networkx as nx
from itertools import product
# from genetic2 import SymbolicRegressor
from genetic import SymbolicRegressor
import sympy as sp
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score
import math




def _calY(f, _x=None, X=None, varNum=None):
    y_pred = []
    len1, len2 = 0, 0
    if _x is None:
        _x = [symbols(f'x{i}') for i in range(500)]
    if X is None:
        # X = _X
        print("X is None")
        len2 = varNum
    else:
        len2 = len(X)
        
    print("len2=",len2)
    len1 = X[0].shape[0]
    print("len1=",len1)
    for i in range(len1):
        _sub = {}
        for j in range(len2):
            _sub.update({_x[j]: X[j][i]})
        y_pred.append(f.evalf(subs=_sub))
    return y_pred


def calculate_nmse(observed, predicted):
    n = len(observed)
    mse = np.mean((observed - predicted) ** 2)
    variance = np.var(observed)
    nmse = mse / variance
    return nmse

def calculate_rmse(observed, predicted):
    n = len(observed)
    mse = np.mean((observed - predicted) ** 2)
    rmse = math.sqrt(mse)
    return rmse




def GBasedSR(_x,X,Y,Gen,SR_method="gplearn"):
    # est_gp = SymbolicRegressor(population_size=1,
    #                        generations=100, stopping_criteria=0.01,
    #                        p_crossover=0, p_subtree_mutation=0.2,
    #                        p_hoist_mutation=0.3, p_point_mutation=0.5,
    #                        max_samples=0.9, verbose=1,
    #                        parsimony_coefficient=0.01, random_state=0)

    est_gp = SymbolicRegressor(population_size=100,
                               generations=200, stopping_criteria=0.01,
                                 p_crossover=0.7, p_subtree_mutation=0.1,
                                    p_hoist_mutation=0.05, p_point_mutation=0.1,
                                    max_samples=0.9, verbose=1,
                                    parsimony_coefficient=0.01, random_state=0)
                            
                                

    est_gp.fit(X, Y)

    best_fitness = est_gp._program.fitness_
    print("best_fitness:",best_fitness)

    best_str = est_gp._program.__str__()
    print("best_str:",best_str)

    y_pred = est_gp.predict(X)

    print("y_pred shape:",y_pred.shape)


    expression = sp.expand(est_gp._program.get_expression())

    print("expression:",expression)



    return y_pred,expression
    

    



def CalTaylorFeaturesGraph(taylor_num,f_taylor,_x,X,Y,population,Gen,Pop,repeatNum,SR_method="Operon"):
    print('In CalTaylorFeaturesGraph')
    # taylor_num.append(0)
    # taylor_num.append(0)
    varNum = X.shape[1]
    k = 2
    G = nx.Graph()
    nodes = [f'x{i}' for i in range(varNum)]
    for i in range(varNum):
        node_name = f'x{i}'
        G.add_node(node_name)
    node_combinations = list(product(nodes, repeat=2))
    # print(node_combinations)

    flag_array = [[z for z in range(2)] for _ in range(varNum)]
    var_array = [[j for j in range(2)] for _ in range(varNum)]

    combinations_flag = list(product(*flag_array))

    Number_record = [combination for combination in combinations_flag if sum(combination) <= k]
    # print(Number_record)
  
    

    fl = [sum(combination) for combination in Number_record]
    # print(fl)
    # 添加连接关系
    print("taylor_num:",taylor_num)
    for i,combination in enumerate(Number_record):
        if fl[i] == 2:
            combin_flag = False
            u = 0
            for j,com_num in enumerate(combination):
                if combin_flag == False:
                    if com_num == 1:
                        combin_flag = True
                        u = j
                    else:
                        pass
                else:
                    if com_num == 1:
                        # print("taylor:",i)
                        if abs(taylor_num[i+1]) > 0.001:
                            print("taylor:",taylor_num[i+1])
                            G.add_edge('x'+str(u), 'x'+str(j))
                            print('x'+str(u), 'x'+str(j))

    print("图的节点集合:", G.nodes)

    subgraphs = list(nx.connected_components(G))
    num_subgraphs = len(subgraphs)

    for subgraph in subgraphs:
        print("子图:",subgraph)

    # print("f_taylor:",f_taylor)


    expression = sp.expand(f_taylor)
    float_values = [term for term in sp.Add.make_args(expression) if isinstance(term, sp.Float)]
    print("float_value:",float_values)
    expression_without_floats = expression - sp.Add(*float_values)
    print("exppp term:",expression_without_floats)
    expend_exp = sp.expand(expression_without_floats)

    all_function = []
    y_all = []
    y_terms = []
    if not all_function:
        all_function.append(sp.Integer(0))
    for subgraph in subgraphs:
        variable_list = [sp.Symbol(var) for var in subgraph]
        # terms_with_subgraph = [term for term in sp.Add.make_args(expression_without_floats) if any(var in term.free_symbols for var in variable_list)]
        # combined_expression = sp.Add(*terms_with_subgraph)
        combined_expression = sum(term for term in expression_without_floats.as_ordered_terms() if any(term.has(var) for var in variable_list))

        print("子表达式含有 x：", combined_expression)
        if len(subgraph) == -1 :
            all_function.append(combined_expression)
            # print("123")

        else:
            # tops_str =TBased_SR(_x, X, Y, population, Gen, Pop, repeatNum,SR_method="Bingo")
            # tops_str = "1"
            y_pred,expression =GBasedSR(_x, X, Y, Gen, SR_method="gplearn")

            # terms = expression.args
            # for term in terms:
            #     print("term:",term)

            expression = sp.expand(expression)

            expr = sp.sympify(expression)
            terms = sp.Add.make_args(expr)
            for term in terms:
                print("term:",term)
                y_pred = _calY(term,_x,X.T,len(_x))

                print("y_pred:",len(y_pred))
                # print("y_pred:",y_pred) 
                y_all.append(y_pred)
                y_terms.append(term)




            
            # y_pred=0 
            # y_all.append(y_pred)

        print("float:",float(float_values[0]))
        y_terms.append(float(float_values[0]))
        print("y_all",len(y_all))
        print("X.shape[0]:",X.shape[0])
        float_values_all = []
        for i in range(X.shape[0]):
            float_values_all.append(float(float_values[0]))

        # float_values_all = float_values * X.shape[0]
        y_all.append(float_values_all)
        print("y_all",len(y_all))


    
        
    lasso_model = Lasso(alpha=0.01)  # alpha是正则化强度，可以调整以控制稀疏性

    # 训练模型
    y_all = np.array(y_all).T
    # print("y_all shape:",y_all.shape)
    # print("y_all shape 1:",y_all.shape[1])
    # print("y_all shape 0:",y_all.shape[0])
    # print("Y shape:",Y.shape)
    y_all_length = y_all.shape[1]
    lasso_model.fit(y_all, Y)

    # 获取稀疏系数
    Taylor_sparse = lasso_model.coef_


    Y_pred = lasso_model.predict(y_all)
    y_pred = Y_pred.reshape((-1, 1))
    print("y_pred shape:",y_pred.shape)
    print("Y shape:",Y.shape)


    mse = mean_squared_error(Y, Y_pred)
    # r2 = 0.0

    r2 = r2_score(Y, Y_pred)

    nmse = calculate_nmse(Y, Y_pred)
    # nmse = mse / np.var(Y)
    # rmse = calculate_rmse(Y, Y_pred)
    rmse = np.sqrt(mse)

    print("mse:",mse)
    print("r2_score:",r2)
    print("nmse:",nmse)
    print("rmse:",rmse)

    new_formulas = []

    for i in range(y_all_length):
        # print("Taylor_sparse:",Taylor_sparse[i])
        # print("y_terms:",y_terms[i])
        new_formula = sp.Mul(Taylor_sparse[i], y_terms[i])
        new_formulas.append(new_formula)


    # equation = sp.expand(expression_without_floats)
    # for i in range(len):
    #     equation 

    combined_formula = sp.Add(*new_formulas)
    # print("combined_formula:",combined_formula)
    print("combined_formula:",str(combined_formula))

# 现在 new_formulas 是一个包含乘法后的公式的列表




    return rmse,None,None,Y_pred
          



    # all_combined_expression = 0
    # for item in all_function:
    #     if isinstance(item, (int, float)):
    #         all_combined_expression += item
    #     elif isinstance(item, list) and all(isinstance(expr, sp.Expr) for expr in item):
    #         all_combined_expression += sum(item)
    # # print("all_function",str(all_function))
    # # all_combined_expression = sp.Add(*all_function)
    # print("all_combined_expression:",all_combined_expression)

    # f_all = sympify(str(all_combined_expression))
    # nmse = 0
    # y_pred_all = None

    # y_pred_all = _calY(f_all,_x,X.T,len(_x))
    # # print("y_pred_all:",y_pred_all)
    # rmse = calculate_nmse(Y, y_pred_all)
    # print("nmse:",rmse)


    # try:
    #     y_pred_all = Metrics2(f_all, _x, X, Y).calY()
    #     nmse = mean_squared_error(Y, y_pred_all)
    #     print("nmse:",nmse)

    # except BaseException:
    #     print("BaseException:Cal Taylor Features Graph")


    # return [rmse],f_all,population,y_pred_all
    
   
def cal_master(filename):
    _init()

    _x = [symbols(f'x{i}') for i in range(500)]
    set_value('_x', _x)
    _x = get_value('_x')

    X_Y = np.loadtxt(filename,dtype=np.float64,skiprows=1)

    X,Y = np.split(X_Y, (-1,), axis=1)

    # X, Y = np.split(X_Y, (-1,), axis=1)

    print("X.shape:",X.shape[0])
    print("Y.shape:",Y.shape[0])

    Metric = []
    linalg = "solve"
    metric = Metrics(varNum=X.shape[1], dataSet=X_Y, linalg=linalg)
    metric.getFlag()

    temp_Y_pred = metric._calY(metric.f_low_taylor)
    print('NMSE of polynomial and lower order polynomial after sorting:', metric.nmse, metric.low_nmse)
    print("f_Taylor: ", metric.f_taylor)

    Generation = 100
    Pop = 100
    repeatNum = 1
    population = None
    end_fitness, programs, population, Y_pred = CalTaylorFeaturesGraph(metric.f_taylor_num,metric.f_taylor, _x[:X.shape[1]], X, Y, population,Generation,Pop, repeatNum, SR_method = "Taylor_Operon")



def cal_x222(filename):
    X_Y = np.loadtxt(filename,dtype=np.float64,skiprows=1)

    X,Y = np.split(X_Y, (-1,), axis=1)

    # X, Y = np.split(X_Y, (-1,), axis=1)

    print("X.shape:",X.shape)
    print("Y.shape:",Y.shape)
    # 构建设计矩阵
    A = np.vstack([X, X**2, X**3]).T

    # 最小二乘法求解
    coefficients, residuals, rank, singular_values = np.linalg.lstsq(A, Y, rcond=None)
    a, b, c = coefficients

    # 预测值
    Y_pred = a * X + b * X**2 + c * X**3

    # 计算 MSE
    mse = mean_squared_error(Y, Y_pred)

    # 计算 R^2
    r2 = r2_score(Y, Y_pred)

    # 打印结果
    print("拟合系数:")
    print("a =", a)
    print("b =", b)
    print("c =", c)
    print("MSE =", mse)
    print("R^2 =", r2)

def cal_x22(filename):
    X_Y = np.loadtxt(filename, dtype=np.float64, skiprows=1)

    X, Y = np.split(X_Y, (-1,), axis=1)

    print("X.shape:", X.shape)
    print("Y.shape:", Y.shape)

    # 构建设计矩阵
    A = np.vstack([X[:, 0], X[:, 0]**2, X[:, 0]**3, X[:, 1], X[:, 1]**2, X[:, 1]**3]).T

    # 最小二乘法求解
    coefficients, residuals, rank, singular_values = np.linalg.lstsq(A, Y, rcond=None)
    a, b, c, d, e, f = coefficients

    # 预测值
    Y_pred = a * X[:, 0] + b * X[:, 0]**2 + c * X[:, 0]**3 + d * X[:, 1] + e * X[:, 1]**2 + f * X[:, 1]**3

    # 计算 MSE
    mse = mean_squared_error(Y, Y_pred)

    # 计算 R^2
    r2 = r2_score(Y, Y_pred)

    # 打印结果
    print("拟合系数:")
    print("a =", a)
    print("b =", b)
    print("c =", c)
    print("d =", d)
    print("e =", e)
    print("f =", f)
    print("MSE =", mse)
    print("R^2 =", r2)

def cal_x212(filename):
    X_Y = np.loadtxt(filename, dtype=np.float64, skiprows=1)

    X, Y = np.split(X_Y, (-1,), axis=1)

    print("X.shape:", X.shape)
    print("Y.shape:", Y.shape)

    # 构建设计矩阵
    A = np.vstack([
        X[:, 0], X[:, 0]**2, X[:, 0]**3, X[:, 0]**4, X[:, 0]**5, X[:, 0]**6,
        X[:, 1], X[:, 1]**2, X[:, 1]**3, X[:, 1]**4, X[:, 1]**5, X[:, 1]**6
    ]).T

    # 最小二乘法求解
    coefficients, residuals, rank, singular_values = np.linalg.lstsq(A, Y, rcond=None)
    a, b, c, d, e, f, g, h, i, j, k, l = coefficients

    # 预测值
    Y_pred = (a * X[:, 0] + b * X[:, 0]**2 + c * X[:, 0]**3 + d * X[:, 0]**4 + e * X[:, 0]**5 + f * X[:, 0]**6 +
              g * X[:, 1] + h * X[:, 1]**2 + i * X[:, 1]**3 + j * X[:, 1]**4 + k * X[:, 1]**5 + l * X[:, 1]**6)

    # 计算 MSE
    mse = mean_squared_error(Y, Y_pred)

    # 计算 R^2
    r2 = r2_score(Y, Y_pred)

    # 打印结果
    print("拟合系数:")
    print("a =", a)
    print("b =", b)
    print("c =", c)
    print("d =", d)
    print("e =", e)
    print("f =", f)
    print("g =", g)
    print("h =", h)
    print("i =", i)
    print("j =", j)
    print("k =", k)
    print("l =", l)
    print("MSE =", mse)
    print("R^2 =", r2)

from sklearn.linear_model import Lasso

def cal_x23(filename):
    X_Y = np.loadtxt(filename, dtype=np.float64, skiprows=1)

    X, Y = np.split(X_Y, (-1,), axis=1)

    print("X.shape:", X.shape)
    print("Y.shape:", Y.shape)

    # 构建设计矩阵
    A = np.vstack([
        X[:, 0], X[:, 0]**2, X[:, 0]**3, X[:, 0]**4, X[:, 0]**5, X[:, 0]**6,
        X[:, 1], X[:, 1]**2, X[:, 1]**3, X[:, 1]**4, X[:, 1]**5, X[:, 1]**6
    ]).T

    # Lasso 回归拟合
    lasso = Lasso(alpha=0.1)  # 设置 alpha 参数
    lasso.fit(A, Y)
    coefficients = lasso.coef_

    # 预测值
    Y_pred = np.dot(A, coefficients)

    # 计算 MSE
    mse = mean_squared_error(Y, Y_pred)

    # 计算 R^2
    r2 = r2_score(Y, Y_pred)

    # 打印结果
    print("拟合系数:")
    for i, coef in enumerate(coefficients):
        print(f"coefficients[{i}] =", coef)
    print("MSE =", mse)
    print("R^2 =", r2)

import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

def polynomial_regression(filename, degree):
    X_Y = np.loadtxt(filename, dtype=np.float64, skiprows=1)

    X, Y = np.split(X_Y, (-1,), axis=1)

    print("X.shape:", X.shape)
    print("Y.shape:", Y.shape)

    # 创建多项式特征
    poly_features = PolynomialFeatures(degree=degree)
    X_poly = poly_features.fit_transform(X)

    # 线性回归拟合
    lin_reg = LinearRegression()
    lin_reg.fit(X_poly, Y)

    # 预测值
    Y_pred = lin_reg.predict(X_poly)

    # 计算 MSE
    mse = mean_squared_error(Y, Y_pred)

    # 计算 R^2
    r2 = r2_score(Y, Y_pred)

    # 打印结果
    print("拟合系数:")
    # for i, coef in enumerate(lin_reg.coef_[0]):
    #     print(f"coefficients[{i}] =", coef)
    print("degree:",degree)
    print("MSE =", mse)
    print("R^2 =", r2)

def cal_x21(filename, degree):
    X_Y = np.loadtxt(filename, dtype=np.float64, skiprows=1)

    X, Y = np.split(X_Y, (-1,), axis=1)

    print("X.shape:", X.shape)
    print("Y.shape:", Y.shape)

    # 生成多项式特征
    X_poly = generate_polynomial_features(X, degree)

    # 线性回归拟合
    model = LinearRegression()
    model.fit(X_poly, Y)

    # 预测值
    Y_pred = model.predict(X_poly)

    # 计算 MSE
    mse = mean_squared_error(Y, Y_pred)

    # 计算 R^2
    r2 = r2_score(Y, Y_pred)

    # 打印结果
    print("拟合系数:")
    print("degree =", degree)
    print("Coefficients:", model.coef_)
    print("Coefficients length :", model.coef_.shape)

    print("MSE =", mse)
    print("R^2 =", r2)

def generate_polynomial_features(X, degree):
    n_samples, n_features = X.shape
    X_poly = np.ones((n_samples, 1))

    for d in range(1, degree + 1):
        for feature in range(n_features):
            X_poly = np.column_stack((X_poly, np.power(X[:, feature], d)))

    return X_poly
if __name__ == '__main__':
    sys.setrecursionlimit(300)
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--fileName', default='example.tsv', type=str)
    args = argparser.parse_args()
    print(args.fileName)
    # cal_master(args.fileName)
    # cal_x2(args.fileName)
    for i in range(1,9):
        cal_master(args.fileName)

    # cal_x2(args.fileName)
        # cal_x21(args.fileName, i)
        # polynomial_regression(args.fileName, i)
    # polynomial_regression(args.fileName, 1)