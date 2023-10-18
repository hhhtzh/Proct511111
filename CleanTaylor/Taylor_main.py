import sys
import argparse
import numpy as np
from sympy import *
from CalTaylor import Metrics
from _global import _init, set_value, get_value
import networkx as nx
from itertools import product
# from genetic2 import SymbolicRegressor
# from genetic import SymbolicRegressor
from gplearn.genetic import SymbolicRegressor
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
import numpy as np


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
    est_gp = SymbolicRegressor(population_size=1,
                           generations=200, stopping_criteria=0.01,
                           p_crossover=0, p_subtree_mutation=0.2,
                           p_hoist_mutation=0.3, p_point_mutation=0.5,
                           max_samples=0.9, verbose=1,
                           parsimony_coefficient=0.01, random_state=0)

    # est_gp = SymbolicRegressor(population_size=100,
    #                            generations=200, stopping_criteria=0.01,
    #                              p_crossover=0.7, p_subtree_mutation=0.1,
    #                                 p_hoist_mutation=0.05, p_point_mutation=0.1,
    #                                 max_samples=0.9, verbose=1,
    #                                 parsimony_coefficient=0.01, random_state=0)
                            
                                

    est_gp.fit(X, Y)

    best_fitness = est_gp._program.raw_fitness_
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
    if not all_function:
        all_function.append(sp.Integer(0))
    for subgraph in subgraphs:
        variable_list = [sp.Symbol(var) for var in subgraph]
        terms_with_subgraph = [term for term in sp.Add.make_args(expression_without_floats) if any(var in term.free_symbols for var in variable_list)]
        combined_expression = sp.Add(*terms_with_subgraph)
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
                # print("y_pred:",y_pred) 
                y_all.append(y_pred)




            
            # y_pred=0 
            y_all.append(y_pred)
            # tops_str =GBasedSR(_x, X, Y, Gen, SR_method="gplearn")
            # tops_str =TBased_SR(_x, X, Y, population, Gen, Pop, repeatNum,SR_method="Taylor_Operon")
            # all_function.append(tops_str[0])
            # print("tops_str_f:",tops_str)
            # # if abs(formula.subs(x, 0)) < 1e-10:
            # #     formula = sp.Integer(0)  
            # all_function.append(tops_str)

    
        
    lasso_model = Lasso(alpha=0.01)  # alpha是正则化强度，可以调整以控制稀疏性

    # 训练模型
    y_all = np.array(y_all).T
    print("y_all shape:",y_all.shape)
    print("Y shape:",Y.shape)
    lasso_model.fit(y_all, Y)

    # 获取稀疏系数
    Taylor_sparse = lasso_model.coef_


    Y_pred = lasso_model.predict(y_all)

    mse = mean_squared_error(Y, Y_pred)
    # r2 = 0.0

    r2 = r2_score(Y, Y_pred)

    # nmse = calculate_nmse(Y, Y_pred)
    # nmse = mse / np.var(Y)
    rmse = calculate_rmse(Y, Y_pred)


    print("mse:",mse)
    print("r2_score:",r2)
    # print("nmse:",nmse)
    print("rmse:",rmse)


    return None,None,None,Y_pred
          



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







    



if __name__ == '__main__':
    sys.setrecursionlimit(300)
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--fileName', default='example.tsv', type=str)
    args = argparser.parse_args()
    print(args.fileName)
    cal_master(args.fileName)