import copy

from keplar.Algorithm.Alg import KeplarBingoAlg
from keplar.Algorithm.operon_Algorithm import OperonAlg
from keplar.operator.composite_operator import CompositeOp, CompositeOpReturn
from keplar.operator.creator import BingoCreator
from keplar.operator.crossover import BingoCrossover, OperonCrossover
from keplar.operator.evaluator import BingoEvaluator, OperonEvaluator
from keplar.operator.mutation import BingoMutation, OperonMutation
from keplar.operator.reinserter import OperonReinserter
from keplar.operator.selector import BingoSelector, OperonSelector
# from .genetic1 import SymbolicRegressor
from gplearn.genetic import SymbolicRegressor

from .calTaylor_GetTaylor import Metrics2  # ,cal_Taylor_features
from .calTaylor_GetTaylor import Metrics
from ._program import print_program
from ._global import set_value, get_value, _init
import numpy as np
from sklearn.metrics import mean_squared_error  # 均方误差
from sklearn.linear_model import LinearRegression
from sympy import *
import sympy as sp
import networkx as nx
from itertools import product
from keplar.translator.translator import trans_taylor_program, bingo_infixstr_to_func,format_taylor,to_gp
from keplar.population.individual import Individual
     
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score
import math



def Taylor_Based_SR(_x, X, Y, qualified_list, eq_write, population, Gen, Pop, repeatNum, low_polynomial,
                    SR_method="gplearn"):
    set_value('TUIHUA_FLAG', False)
    Y_pred = None
    f_low_taylor, f_low_taylor_mse = qualified_list[-5], qualified_list[-4]
    if low_polynomial == False:
        if SR_method == "gplearn":
            function_set = ['add', 'sub', 'mul', 'div', 'sin', 'cos', 'log', 'exp', 'sqrt']
            est_gp = SymbolicRegressor(population_size=Pop, init_depth=(2, 5),
                                       generations=Gen, stopping_criteria=1e-5, function_set=function_set,
                                       p_crossover=0.7, p_subtree_mutation=0.,
                                       p_hoist_mutation=0., p_point_mutation=0.2,
                                       max_samples=1.0, verbose=1,
                                       parsimony_coefficient=0.1,
                                       n_jobs=1,  #
                                       const_range=(-1, 1),
                                       random_state=repeatNum, low_memory=True)
            print(qualified_list)
            tops_fit, tops_str, population, Y_pred = est_gp.fit(X, Y, qualified_list,
                                                                population_input=population)  # 种群演化返回的list of适应度 和公式
            if est_gp._program.raw_fitness_ > f_low_taylor_mse:
                print(f_low_taylor, f_low_taylor_mse, sep='\n')
                tops_fit.insert(0, f_low_taylor_mse)
                tops_str.insert(0, f_low_taylor)
                Y_pred = "f_low_taylor"
                return tops_fit, tops_str, population, Y_pred
            else:
                return [f_low_taylor_mse], [f_low_taylor], None, Y_pred
        elif SR_method == "Bingo":
            operators = ['+', '-', '*', '/', 'sin', 'cos', 'log', 'exp', 'sqrt']
            x = X
            y = Y
            creator = BingoCreator(50, operators, x, 10, "Bingo")
            evaluator = BingoEvaluator(x, "exp", "lm", "Bingo", y, metric="mse")
            crossover = BingoCrossover("Bingo")
            mutation = BingoMutation(x, operators, "Bingo")
            selector = BingoSelector(0.5, "tournament", "Bingo")
            gen_up_oplist = CompositeOp([crossover, mutation])
            gen_down_oplist = CompositeOpReturn([selector])
            gen_eva_oplist = CompositeOp([evaluator])
            population = creator.do()
            bgsr = KeplarBingoAlg(1000, gen_up_oplist, gen_down_oplist, gen_eva_oplist, 0.001, population)
            bgsr.run()
            tops_fit = population.get_best_fitness()
            tops_str = str(population.get_tar_best())
            population = None
            if tops_fit > f_low_taylor_mse:
                print(f_low_taylor, f_low_taylor_mse, sep='\n')
                Y_pred = "f_low_taylor"
                return [tops_fit], [tops_str], population, Y_pred
            else:
                return [tops_fit], [print_program(tops_str, qualified_list, X, _x)]
        elif SR_method == "Operon":
            x = X
            y = Y
            selector = OperonSelector(5)
            evaluator = OperonEvaluator("MSE", x, y, 0.5, True, "Operon")
            crossover = OperonCrossover(x, y, "Operon")
            mutation = OperonMutation(1, 1, 1, 0.5, x, y, 10, 50, "balanced", "Operon")
            reinsert = OperonReinserter(None, "ReplaceWorst", 10, "Operon", x, y)
            op_up_list = [mutation, crossover]
            op_down_list = [reinsert]
            eva_list = [evaluator]
            op_alg = OperonAlg(1000, op_up_list, op_down_list, eva_list, selector, 1e-5, 1000, 16, x, y)
            op_alg.run()
            tops_fit = op_alg.model_fit
            tops_str = str(op_alg.model_string)
            population = None
            if tops_fit > f_low_taylor_mse:
                print(f_low_taylor, f_low_taylor_mse, sep='\n')
                Y_pred = "f_low_taylor"
                return [tops_fit], [tops_str], population, Y_pred
            else:
                return [tops_fit], [print_program(tops_str, qualified_list, X, _x)]

        else:
            raise ValueError("其他回归暂时未设置")
    else:
        return [f_low_taylor_mse], [f_low_taylor], None, Y_pred


def TBased_SR(_x, X, Y, population, Gen, Pop, repeatNum,SR_method="Operon"):
    set_value('TUIHUA_FLAG', False)
    Y_pred = None
    # f_low_taylor, f_low_taylor_mse = qualified_list[-5], qualified_list[-4]
    # if low_polynomial == False:
    if SR_method == "T_gplearn":
            function_set = ['add', 'sub', 'mul', 'div', 'sin', 'cos', 'log', 'exp', 'sqrt']
            est_gp = SymbolicRegressor(population_size=Pop, init_depth=(2, 5),
                                       generations=Gen, stopping_criteria=1e-5, function_set=function_set,
                                       p_crossover=0.7, p_subtree_mutation=0.,
                                       p_hoist_mutation=0., p_point_mutation=0.2,
                                       max_samples=1.0, verbose=1,
                                       parsimony_coefficient=0.1,
                                       n_jobs=1,  #
                                       const_range=(-1, 1),
                                       random_state=repeatNum, low_memory=True)
            tt = est_gp.fit(X, Y)
            print("tt:",str(tt._best_programs))

            print("done")

            return tt,tt._best_programs
            # print(qualified_list)
            # qualified_list = None
            # tops_fit, tops_str, population, Y_pred = est_gp.fit(X, Y, qualified_list,
            #                                                     population_input=population)  # 种群演化返回的list of适应度 和公式
            # if est_gp._program.raw_fitness_ > f_low_taylor_mse:
            #     print(f_low_taylor, f_low_taylor_mse, sep='\n')
            #     tops_fit.insert(0, f_low_taylor_mse)
            #     tops_str.insert(0, f_low_taylor)
            #     Y_pred = "f_low_taylor"
            #     return tops_fit, tops_str, population, Y_pred
            # else:
            #     return [f_low_taylor_mse], [f_low_taylor], None, Y_pred
    if SR_method == "gplearn":
        function_set = ['add', 'sub', 'mul', 'div', 'sin', 'cos', 'log', 'exp', 'sqrt']
        est_gp = SymbolicRegressor(population_size=Pop, init_depth=(2, 5),
                                    generations=Gen, stopping_criteria=1e-5, function_set=function_set,
                                    p_crossover=0.7, p_subtree_mutation=0.,
                                    p_hoist_mutation=0., p_point_mutation=0.2,
                                    max_samples=1.0, verbose=1,
                                    parsimony_coefficient=0.1,
                                    n_jobs=1,  #
                                    const_range=(-1, 1),
                                    random_state=repeatNum, low_memory=True)
        # print(qualified_list)
        qualified_list = []
        tops_fit, tops_str, population, Y_pred = est_gp.fit(X, Y, qualified_list,
                                                            population_input=population)  # 种群演化返回的list of适应度 和公式
        
        # if est_gp._program.raw_fitness_ > f_low_taylor_mse:
        #     print(f_low_taylor, f_low_taylor_mse, sep='\n')
        #     tops_fit.insert(0, f_low_taylor_mse)
        #     tops_str.insert(0, f_low_taylor)
        #     Y_pred = "f_low_taylor"
        #     return tops_fit, tops_str, population, Y_pred
        # else:
        #     return [f_low_taylor_mse], [f_low_taylor], None, Y_pred
    elif SR_method == "TBingo":
        pass
        
    elif SR_method == "Bingo":
        operators = ['+', '-', '*', '/', 'sin', 'cos', 'log', 'exp', 'sqrt']
        x = X
        y = Y
        creator = BingoCreator(50, operators, x, 10, "Bingo")
        evaluator = BingoEvaluator(x, "exp", "lm", "Bingo", y, metric="mse")
        crossover = BingoCrossover("Bingo")
        mutation = BingoMutation(x, operators, "Bingo")
        selector = BingoSelector(0.5, "tournament", "Bingo")
        gen_up_oplist = CompositeOp([crossover, mutation])
        gen_down_oplist = CompositeOpReturn([selector])
        gen_eva_oplist = CompositeOp([evaluator])
        population = creator.do()
        bgsr = KeplarBingoAlg(300, gen_up_oplist, gen_down_oplist, gen_eva_oplist, 0.001, population)
        bgsr.run()
        tops_fit = population.get_best_fitness()

        y_pred = bgsr.predict(X)
        print("y_pred:",y_pred)
        print("tops_fit:",tops_fit)
        # tops_str = population.target_pop_list[population.get_tar_best()]
        # print("tops_str:",str(tops_str))
        # tops_str1 = population.pop_list[population.get_best()]
        expression_str = bgsr.best_indi
        print("tops_str1:",expression_str)
        # variables_X = [symbols(f'X_{i}') for i in range(0, 100)]

        # expression = sympify(expression_str)
        # substitutions = {X[i]: _x[i] for i in range(len(_x))}
        # top = expression.subs(substitutions)
        # for i in range(len(_x)):
        #     expression = expression.subs(sp.symbols(f'X_{i}'), _x[i])
        # top = expression
        # print("top:",str(top))
        # expression_str = expression_str.replace('X_', 'x')
        # symbols(_x)
        # sympy_expression = sympify(expression_str)
        # # sympy_expression = sp.parse_expr(expression_str)
        # print("sympy_expression:",str(sympy_expression))
        
        # print("tops_str1:",str(tops_str1))
        print("_x:",_x)

        variables = [_x[i] for i in range(len(_x))]
        # variables = [sp.symbols(_x[i]) for i in range(len(_x))]
        print("variables:",variables)

        expression_str = expression_str.replace("X_", "x")

        expression_str = expression_str.replace(")(", ")*(")
        print("expression_str:",expression_str)
        sympy_expression = sp.sympify(expression_str)
        # print("sympy_expression:",str(sympy_expression))
        return [sympy_expression]




        # func,const_arr = bingo_infixstr_to_func(str(expression_str))
        # kep_ind=Individual(func,const_arr)

        # T_program = trans_taylor_program(kep_ind)
        # print("T_program:",str(T_program))
        # print("kep_ind list:",kep_ind.func)
        # print("kep_ind:",kep_ind.const_array)
        # print("kep_ind:",str(kep_ind.equation))
        # print("kep_ind:",kep_ind.format())
        # top = to_gp(kep_ind)
        # print("tops:",str(top))

        # expression = sp.expand(kep_ind.format())
        # expression_str = kep_ind.format()
        # print("expression_str:",expression_str)
        # expression = sympify(expression_str)
        # print("expression:",str(expression))
        # variables_X = [symbols(f'X_{i}') for i in range(0, 100)]

        # 创建一个示例表达式，例如 X_1 + X_2 + ... + X_100
        # expression = sum(variables_X)
        # print("_x",_x)

        # 创建替换映射字典
        # substitutions = {X[i]: _x[i] for i in range(len(_x))}
        # print("substitutions",substitutions)

        # 使用 subs 方法将变量替换
        # top = expression.subs(substitutions)
        # for i in range(len(_x)):
        #     expression = expression.subs(sp.symbols(f'X_{i}'), _x[i])
        # top = expression
        # new_expression = expression.subs(x_1, x1)

        # print("kepler:",kep_ind.format())
        # print("kep_ind:",kep_ind)
        # str_gp = str(kep_ind.format())
        # top = str_gp.replace('X_', 'x')

        # top = simplify(top)
        # top_taylor = trans_taylor_program(kep_ind)
        # print("top:",top_taylor)

        # top = format_taylor(top_taylor,kep_ind.const_array)
        # print("top:",top)
        # print("top:",top)
        # population = None
        # if tops_fit > f_low_taylor_mse:
        #     print(f_low_taylor, f_low_taylor_mse, sep='\n')
        #     Y_pred = "f_low_taylor"
        # return [tops_fit], [top], population, Y_pred
        # else:
        #     return [tops_fit], [print_program(tops_str, qualified_list, X, _x)]
    elif SR_method == "Operon":
        x = X
        y = Y
        # selector = OperonSelector(5)
        evaluator = OperonEvaluator("MSE", x, y, 0.5, True, "Operon")
        crossover = OperonCrossover(x, y, "Operon")
        mutation = OperonMutation(1, 1, 1, 0.5, x, y, 10, 50, "balanced", "Operon")
        reinsert = OperonReinserter(None, "ReplaceWorst", 10, "Operon", x, y)
        op_up_list = [mutation, crossover]
        op_down_list = [reinsert]
        eva_list = [evaluator]
        op_alg = OperonAlg(1000, op_up_list, op_down_list, eva_list, 1e-5, 1000, 16, x, y)
        op_alg.run()
        tops_fit = op_alg.model_fit
        tops_str = str(op_alg.model_string)
        population = None
        # if tops_fit > f_low_taylor_mse:
        #     print(f_low_taylor, f_low_taylor_mse, sep='\n')
        #     Y_pred = "f_low_taylor"
        return [tops_fit], [tops_str], population, Y_pred
        # else:
        #     pass
            # return [tops_fit], [print_program(tops_str, qualified_list, X, _x)]
    elif SR_method == "Taylor_Operon":
        x = X
        y = Y
        data = None
        x_shape = np.shape(x[0])[0]
        evaluator = OperonEvaluator("RMSE", x, y, 0.7, True, "Operon")
        crossover = OperonCrossover(x, y, "Operon")
        mutation = OperonMutation(1, 1, 1, 0.5, x, y, 10, 50, "balanced", "Operon")
        reinsert = OperonReinserter(None, "ReplaceWorst", 10, "Operon", x, y)
        op_up_list = [mutation, crossover]
        op_down_list = [reinsert]
        eva_list = [evaluator]
        op_alg = OperonAlg(1000, op_up_list, op_down_list, eva_list,  1e-5, 128, 16, x, y, data, x_shape)
        for i in range(1):
            op_alg.run()
        
        tops_fit = op_alg.best_fit
        tops_str = str(op_alg.model_string)
        print("tops_str",tops_str)

        expression_str = tops_str.replace('^', '**')
        expression_str = tops_str.replace('X', 'x')

        print("expression_str:",expression_str) 

        # sympy_expression = sp.parse_expr(expression_str)
        # print("sympy_expression:",str(sympy_expression))
        # expression = sympify(expression_str)
        tops_str = expression_str


        print("best_fit:",tops_fit)
        print("tops_str:",expression_str)
        variables = [_x[i] for i in range(len(_x))]
        sympy_expression = sp.sympify(expression_str)



        return [sympy_expression]


    else:
        raise ValueError("其他回归暂时未设置")

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

def calculate_rmse(observed, predicted):
    n = len(observed)
    mse = np.mean((observed - predicted) ** 2)
    rmse = math.sqrt(mse)
    return rmse

def root_mean_squared_error(y_true, y_pred):
    # 计算均方误差 (MSE)
    mse = np.mean((y_true - y_pred)**2)
    
    # 计算 RMSE（均方根误差）
    rmse = np.sqrt(mse)
    
    return rmse

def CalTaylorFeaturesGraph_detelect(taylor_num,f_taylor,_x,X,Y,population,Gen,Pop,repeatNum,SR_method="Operon"):
    print('In CalTaylorFeaturesGraph')
    taylor_num.append(0)
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

        else:
            tops_str =TBased_SR(_x, X, Y, population, Gen, Pop, repeatNum,SR_method="Bingo")
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
    rmse = mean_squared_error(Y, y_pred_all)
    print("nmse:",rmse)


    # try:
    #     y_pred_all = Metrics2(f_all, _x, X, Y).calY()
    #     nmse = mean_squared_error(Y, y_pred_all)
    #     print("nmse:",nmse)

    # except BaseException:
    #     print("BaseException:Cal Taylor Features Graph")


    return [rmse],f_all,population,y_pred_all
    

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
    taylor_num.append(0)
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

    print("Graph f_taylor:",f_taylor)


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
    if len(subgraphs) == 1:
        y_pred,expression =GBasedSR(_x, X, Y, Gen, SR_method="gplearn")
        rmse = calculate_rmse(Y, y_pred)
        return [rmse],[expression],None,y_pred

    else:
        for subgraph in subgraphs:
            variable_list = [sp.Symbol(var) for var in subgraph]
            # terms_with_subgraph = [term for term in sp.Add.make_args(expression_without_floats) if any(var in term.free_symbols for var in variable_list)]
            # combined_expression = sp.Add(*terms_with_subgraph)
            combined_expression = sum(term for term in expression_without_floats.as_ordered_terms() if any(term.has(var) for var in variable_list))
            print("子表达式含有 x：", combined_expression)

            expression_terms = sp.sympify(combined_expression)
            y_pred_terms = _calY(expression_terms,_x,X.T,len(_x))

            if len(subgraph) == -1 :
                all_function.append(combined_expression)

            else:
                # tops_str =TBased_SR(_x, X, Y, population, Gen, Pop, repeatNum,SR_method="Bingo")
                # tops_str = "1"

                y_pred,expression =GBasedSR(_x, X, y_pred_terms, Gen, SR_method="gplearn")

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
                    y_terms.append(term)





                
                # y_pred=0 
                # y_all.append(y_pred)
                # tops_str =GBasedSR(_x, X, Y, Gen, SR_method="gplearn")
                # tops_str =TBased_SR(_x, X, Y, population, Gen, Pop, repeatNum,SR_method="Taylor_Operon")
                # all_function.append(tops_str[0])
                # print("tops_str_f:",tops_str)
                # # if abs(formula.subs(x, 0)) < 1e-10:
                # #     formula = sp.Integer(0)  
                # all_function.append(tops_str)

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

        # print("Y shape:",Y.shape)
        y_all_length = y_all.shape[1]

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

        str_combined_formula = str(combined_formula)

        # equ = 


        return [rmse],[str_combined_formula],None,Y_pred
          


def OriginalTaylorGP(X_Y, Y_pred, population, repeatNum, Generation, Pop, rmseFlag=False, qualified_list=None,
                     SR_method="gplearn"):
    """
    原始版本的TaylorGP
    Args:
        X_Y: 原始数据集或者经过分割后的子数据集
        eq_write: 输出文件

    Returns: top,保存每个子块的topk个体数组
    :param SR_method:

    """
    # print('In OriginalTaylorGP')
    _init()
    # x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24, x25, x26, x27, x28, x29 = symbols(
    #     "x0,x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16,x17,x18,x19,x20,x21,x22,x23,x24,x25,x26,x27,x28,x29 ")

    # set_value('_x',
    #           [x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22,
    #            x23,
    #            x24, x25, x26, x27, x28, x29])
    
    _x = [symbols(f'x{i}') for i in range(500)]
    set_value('_x', _x)
    _x = get_value('_x')
    eqName = 'SubTaylorGP.out'  # eqName = fileName[:-4]+'.out' 原先的打印都是针对整体的，所以此处应该都用不上了
    eq_write = open(eqName, "w+")
    eq_write.write(
        'Gen|Top1|Length1|Fitness1|Top2|Length2|Fitness2|Top3|Length3|Fitness3|Top4|Length4|Fitness4|Top5|Length5|Fitness5|Top6|Length6|Fitness6|Top7|Length7|Fitness7|Top8|Length8|Fitness8|Top9|Length9|Fitness9|Top10|Length10|Fitness10\n')

    X, Y = np.split(X_Y, (-1,), axis=1)
    end_fitness, program, findBestFlag, Y_pred, temp_Y_pred = None, None, False, None, copy.deepcopy(Y_pred)
    if qualified_list == None:  # 保证泰勒多项式不用重复计算
        Metric = []     
        # for linalg in ["solve", "lstsq"]:
        #     loopNum = 0
        #     while True:
        #         metric = Metrics(varNum=X.shape[1], dataSet=X_Y, linalg=linalg)
        #         loopNum += 1
        #         Metric.append(metric)
        #         if loopNum == 5 and X.shape[1] <= 2:
        #             break
        #         elif loopNum == 4 and (X.shape[1] > 2 and X.shape[1] <= 3):
        #             break
        #         elif loopNum == 3 and (X.shape[1] > 3 and X.shape[1] <= 4):
        #             break
        #         elif loopNum == 2 and (X.shape[1] > 4):
        #             break
        linalg = "solve"

        metric = Metrics(varNum=X.shape[1], dataSet=X_Y, linalg=linalg)

        metric.getFlag()
        # Metric.sort(key=lambda x: x.low_nmse)
        # metric = Metric[0]
        temp_Y_pred = metric._calY(metric.f_low_taylor)
        print('NMSE of polynomial and lower order polynomial after sorting:', metric.nmse, metric.low_nmse)
        print("f_Taylor: ", metric.f_taylor)
        eq_write.write(str(-1) + '|' + str(metric.f_low_taylor) + '|' + '10' + '|' + str(metric.low_nmse) + '|' + '\n')
        # if metric.nmse < 0.01:
        #     metric.nihe_flag = True
        # else:
        print("call  Linear regression to change nmse and f_taylor")
        lr_est = LinearRegression().fit(X, Y)
        print('coef: ', lr_est.coef_)
        print('intercept: ', lr_est.intercept_)
        lr_Y_pred = lr_est.predict(X)
        lr_nmse = mean_squared_error(lr_Y_pred, Y)

        f = str(lr_est.intercept_[0])
        for i in range(X.shape[1]):
            if lr_est.coef_[0][i] >= 0:
                f += '+' + str(lr_est.coef_[0][i]) + '*x' + str(i)
            else:
                f += str(lr_est.coef_[0][i]) + '*x' + str(i)
        print("f_lr and nmse_lr" + f + "  " + str(lr_nmse))
        '''
        fitness = mean_squared_error(lr_est.predict(test_X), test_y, squared=False)  # RMSE
        print('LR_predict_fitness: ', fitness)                
        '''
        print("f_tailor L:",metric.f_taylor)

        for i in range(1):
            if lr_nmse < metric.nmse:
                metric.nmse = lr_nmse
                metric.f_taylor = sympify(f)
                metric.bias = 0.
            if metric.nmse < 0.01:
                metric.nihe_flag = True  # 奇偶性判断前需要先拟合
            if lr_nmse < metric.low_nmse:
                metric.low_nmse = lr_nmse
                metric.f_low_taylor = sympify(f)
                temp_Y_pred = lr_Y_pred
            if rmseFlag == True: return metric.nmse
            # time_end2 = time()
            # print('Pretreatment_time_cost', (time_end2 - time_start2) / 3600, 'hour')
            # self.global_fitness, self.sympy_global_best = metric.low_nmse, metric.f_low_taylor
            # if metric.nmse < 0.1:
            #     metric.nihe_flag = True
            # else:
            #     metric.bias = 0.
            #     print('Fitting failed')
            if metric.low_nmse < 1e-5:
                end_fitness, programs, Y_pred = [metric.low_nmse], [metric.f_low_taylor], Y
            else:
                qualified_list = []
                qualified_list.extend(
                    [metric.judge_Bound(),
                    metric.f_low_taylor,
                    metric.low_nmse,
                    metric.bias,
                    metric.judge_parity(),
                    metric.judge_monotonicity()])
                # print(qualified_list)

            # elif metric.nihe_flag and (metric.judge_additi_separability() or metric.judge_multi_separability()):
            #     end_fitness, programs,population = CalTaylorFeatures(metric.f_taylor, _x[:X.shape[1]], X, Y, population,Generation,Pop, repeatNum, eq_write)
            if end_fitness is None:
                
                # end_fitness, programs, population, Y_pred = Taylor_Based_SR(_x, X, change_Y(Y, qualified_list), qualified_list,
                #                                                             eq_write, population, Generation, Pop, repeatNum,
                #                                                             qualified_list[2] < 1e-5, SR_method=SR_method)
                # if isinstance(Y_pred,str):
                #     Y_pred = temp_Y_pred
                # # Y_pred = programs[0].predict(X)
                # print(end_fitness)
                # print(programs)
                # print('fitness_and_program', end_fitness[0], programs[0], sep='\n')
                print("f_tailor G:",metric.f_taylor)
                end_fitness, programs, population, Y_pred = CalTaylorFeaturesGraph(metric.f_taylor_num,metric.f_taylor, _x[:X.shape[1]], X, Y, population,Generation,Pop, repeatNum, SR_method = "Taylor_Operon")
            
            if end_fitness[0] < 1e-5: findBestFlag = true

            return [end_fitness, programs, population, findBestFlag, qualified_list, Y_pred]
        

        
    


def change_Y(Y, qualified_list):
    if Y is None:
        return None
    if qualified_list[4] != -1:  # .parity_flag:
        if abs(qualified_list[3]) > 1e-5:  # abs(self.bias) > 1e-5:
            return Y - qualified_list[3]  # Y - self.bias
    if qualified_list[5] == 2:  # self.di_jian_flag
        return Y * (-1)
    else:
        return Y
