import sys
import argparse
import numpy as np
from sympy import *
from CalTaylor import Metrics
from _global import _init, set_value, get_value
import networkx as nx
from itertools import product
# from gplearn.genetic import SymbolicRegressor, BaseSymbolic, MAX_INT


# import sympy as sp
# from ..operator.mutation import BingoMutation
# from ..keplar.operator.creator import BingoCreator
# from Kepler.keplar.operator.mutation import BingoMutation, OperonMutation
# from ..Kepler.keplar.Algorithm.Alg import KeplarBingoAlg
# from ..Kepler.keplar.operator.composite_operator import CompositeOp, CompositeOpReturn

# from ..Kepler.keplar.operator.evaluator import BingoEvaluator, OperonEvaluator
# from ..Kepler.keplar.operator.reinserter import OperonReinserter
# from ..Kepler.keplar.operator.selector import BingoSelector, OperonSelector
# from .genetic1 import SymbolicRegressor
# from gplearn.genetic import SymbolicRegressor


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

def calculate_nmse(true_values, predicted_values):
    """
    计算NMSE（Normalized Mean Squared Error）的函数

    参数:
    true_values (numpy数组): 真实的目标值
    predicted_values (numpy数组): 模型的预测值

    返回:
    nmse (float): 计算得到的NMSE值
    """

    # 计算均方误差（MSE）
    mse = np.mean((true_values - predicted_values) ** 2)

    # 计算目标值的方差
    variance = np.var(true_values)

    # 计算NMSE
    nmse = mse / variance

    return nmse

# # 示例用法
# true_values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
# predicted_values = np.array([1.1, 2.2, 2.8, 4.2, 5.1])
# nmse = calculate_nmse(true_values, predicted_values)
# print("NMSE:", nmse)

def root_mean_squared_error(y_true, y_pred):
    # 计算均方误差 (MSE)
    mse = np.mean((y_true - y_pred)**2)
    
    # 计算 RMSE（均方根误差）
    rmse = np.sqrt(mse)
    
    return rmse

# def TBasedSR(_x,X,Y,Gen,SR_method="Bingo"):
#     operators = ['+', '-', '*', '/', 'sin', 'cos', 'log', 'exp', 'sqrt']
#     x = X
#     y = Y
#     creator = BingoCreator(1, operators, x, 10, "Bingo")
#     evaluator = BingoEvaluator(x, "exp", "lm", "Bingo", y, metric="mse")

#     mutation = BingoMutation(x, operators, "Bingo")
#     selector = BingoSelector(0.5, "tournament", "Bingo")

#     population = creator.do()
#     gen_up_oplist =  CompositeOp([mutation])
#     gen_down_oplist = CompositeOpReturn([selector])
#     gen_eva_oplist = CompositeOp([evaluator])


#     # gen_down_oplist = CompositeOp([mutation])

#     bgsr = KeplarBingoAlg(1000, gen_up_oplist, gen_down_oplist, gen_eva_oplist, 0.001, population)
#     bgsr.do()
#     tops_fit = population.get_best_fitness()
#     tops_str = str(population.get_tar_best())
#     print("tops_str:",tops_str)
#     # population = None

def GBasedSR(_x,X,Y,Gen,SR_method="gplearn"):
    



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
    if not all_function:
        all_function.append(sp.Integer(0))
    for subgraph in subgraphs:
        variable_list = [sp.Symbol(var) for var in subgraph]
        terms_with_subgraph = [term for term in sp.Add.make_args(expression_without_floats) if any(var in term.free_symbols for var in variable_list)]
        combined_expression = sp.Add(*terms_with_subgraph)
        print("子表达式含有 x：", combined_expression)
        if len(subgraph) == 1 :
            all_function.append(combined_expression)
            # print("123")

        else:
            # tops_str =TBased_SR(_x, X, Y, population, Gen, Pop, repeatNum,SR_method="Bingo")
            tops_str = "1"
            # tops_str =TBased_SR(_x, X, Y, population, Gen, Pop, repeatNum,SR_method="Taylor_Operon")
            # all_function.append(tops_str[0])
            print("tops_str_f:",tops_str)
            # if abs(formula.subs(x, 0)) < 1e-10:
            #     formula = sp.Integer(0)  
            all_function.append(tops_str)
    all_combined_expression = 0
    for item in all_function:
        if isinstance(item, (int, float)):
            all_combined_expression += item
        elif isinstance(item, list) and all(isinstance(expr, sp.Expr) for expr in item):
            all_combined_expression += sum(item)
    # print("all_function",str(all_function))
    # all_combined_expression = sp.Add(*all_function)
    print("all_combined_expression:",all_combined_expression)

    f_all = sympify(str(all_combined_expression))
    nmse = 0
    y_pred_all = None

    y_pred_all = _calY(f_all,_x,X.T,len(_x))
    # print("y_pred_all:",y_pred_all)
    rmse = calculate_nmse(Y, y_pred_all)
    print("nmse:",rmse)


    # try:
    #     y_pred_all = Metrics2(f_all, _x, X, Y).calY()
    #     nmse = mean_squared_error(Y, y_pred_all)
    #     print("nmse:",nmse)

    # except BaseException:
    #     print("BaseException:Cal Taylor Features Graph")


    return [rmse],f_all,population,y_pred_all
    
   
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