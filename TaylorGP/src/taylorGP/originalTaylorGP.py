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
from .genetic1 import SymbolicRegressor
from .calTaylor import Metrics2  # ,cal_Taylor_features
from .calTaylor1 import Metrics
from ._program import print_program
from ._global import set_value, get_value, _init
import numpy as np
from sklearn.metrics import mean_squared_error  # 均方误差
from sklearn.linear_model import LinearRegression
from sympy import *

from sympy import sympify, symbols, Add

def collect_terms(expression_str, _x, threshold=0.001):
    # 使用 sympify 将字符串表达式转换为 Sympy 表达式
    expression = sympify(expression_str)

    # 解析额外的变量 _x 中包含的变量
    extra_vars = symbols(_x)

    # 分割表达式为多个项
    terms = expression.as_ordered_terms()

    # 初始化用于存放仅含有 _x 中变量的项的字典
    x_terms_dict = {str(x): [] for x in extra_vars}

    for term in terms:
        # 获取系数
        coeff = term.as_coefficients_dict()
        all_vars = term.free_symbols

        # 检查项是否只含有一个 _x 中的变量，并且所有变量在 extra_vars 中
        if len(all_vars) == 1 and all_vars.intersection(extra_vars):
            # 去除系数小于 threshold 的项
            if abs(coeff[1]) >= threshold:
                var_name = str(all_vars.pop())
                x_terms_dict[var_name].append(term)

    # 将仅含有 _x 中各个变量的项分别组合成公式
    x_expressions = {var_name: Add(*terms_list).simplify() for var_name, terms_list in x_terms_dict.items()}

    return x_expressions

def filter_equation(equation_string):
    # 将方程按加号分割成项的列表
    terms = equation_string.split(" + ")

    # 初始化一个空列表，用于存储满足条件的项
    filtered_terms = []

    # 遍历每一项，并检查系数是否小于0.01
    for term in terms:
        # 寻找"x"的位置，将系数和变量分割开
        x_index = term.find("*x")
        if x_index == -1:  # 没有找到"x"，说明是常数项
            coefficient_str = term
            variable = ""
        else:
            coefficient_str = term[:x_index]
            variable = term[x_index:]

        # 处理系数
        coefficient = float(coefficient_str) if coefficient_str else 1.0

        # 将系数小于0.01的项添加到新的列表中
        if abs(coefficient) < 0.01:
            continue
        filtered_terms.append(term)

    # 生成新的方程
    new_equation = " + ".join(filtered_terms)

    return new_equation

def split_formula(formula):
    formula = formula.replace(" ", "")
    formula = "+" + formula.strip()

    terms = []
    term = ""
    operator = "+"

    for i in range(len(formula)):
        char = formula[i]
        if char == "+" or char == "-":
            if term != "":
                terms.append(operator + term)
            term = ""
            operator = char
        else:
            term += char

    if term != "":
        terms.append(operator + term)

    return terms


def assign_terms_to_equations(terms, subsets):
    equations = {}
    equations[0] = [] # 用于存储不属于任何子集的项

    for term in terms:
        subsets_count = 0
        subset_index = -1
        is_exclusive = False
        is_inclusive = True

        for i, subset in enumerate(subsets):
            equations[i] = []
            if any(element in term for element in subset):
                subsets_count += 1
                subset_index = i
                is_inclusive = False


        if subsets_count == 1:
            is_exclusive = True



        for i, subset in enumerate(subsets):
            if i != subset_index:
                if any(element in term for element in subset):
                    is_exclusive = False
                    break

        if is_exclusive:
            if subset_index not in equations:
                equations[subset_index] = []
            
            equations[subset_index].append(term)

        if is_inclusive:
            equations[0].append(term)

    formula = ""

    for subset_index, equation_terms in equations.items():
        equation = "".join(equation_terms)
        # print("subset_index",subset_index)
        formula += equation + "\n"
        # formula += equation + "\n" if equation else " \n" 

    # for subset_index in range(len(subsets)):

    # for i, subset in enumerate(subsets):
    #     # print("fff",i)
    #     if equations[i]:
    #         equation = "".join(equations[i])
    #     else:
    #         equation = None
    #     formula += equation + "\n" if equation else "NONE\n" 




    return formula.strip()

def CalTaylorFeaturesGraph(taylor_num,f_taylor, _x, X, Y, qualified_list, eq_write, population, Gen, Pop, repeatNum, low_polynomial,
                    SR_method="gplearn"):
    print('In CalTaylorFeaturesGraph')
    G = nx.Graph()
    
    metric = Metrics2(f_taylor,_x,X,Y)
    if metric.judge_Low_polynomial():
        return [metric.low_nmse], [metric.f_low_taylor],None
    
    #我的python也还不熟练，暂时用简易易理解的方式实现
    if X.shape[1]>1:
        k=2
        if X.shape[1] == 2:
            G.add_nodes_from(['x0', 'x1'])

            # 添加连接关系
            num = 0
            for x_0 in range(2):
                x_1 = k - x_0
                num+=1
                #x_0
                if all(x in [0, 1] for x in [x_0, x_1]):
                    
                    if abs(taylor_num[num]) >= 0.01:
                        if x_0 == 1 and x_1 == 1:
                            G.add_edge('x0', 'x1')

        if X.shape[1] == 3:
            # 添加连接关系
            G.add_nodes_from(['x0', 'x1', 'x2'])
            num = 0
            for x_0 in range(2):
                for x_1 in range(2):
                    x_2 = k - x_0 - x_1
                    #使用num判断taylor_num的哪一个值
                    num+=1
                    '''
                    根据前面的判断,只有当x_0,x_1,x_2都是0或者1的时候才会添加边,这个边代表着这三个变量的关系,且因为k=2
                    所以有且仅有两个变量为1,其余变量为0,这两个变量之间的关系就是这个边的关系
                    '''
                    if all(x in [0, 1] for x in [x_0, x_1, x_2]):
                        #判断taylor_num的值是否小于阈值,如果小于阈值,则添加边
                        if abs(taylor_num[num]) >= 0.01:
                            if x_0 == 1 and x_1 == 1:
                                G.add_edge('x0', 'x1')
                            if x_0 == 1 and x_2 == 1:
                                G.add_edge('x0', 'x2')
                            if x_1 == 1 and x_2 == 1:
                                G.add_edge('x1', 'x2')

        if X.shape[1] == 4:
            G.add_nodes_from(['x0', 'x1', 'x2', 'x3'])

            # 添加连接关系
            num = 0
            for x_0 in range(2):
                for x_1 in range(2):
                    for x_2 in range(2):
                        x_3 = k - x_0 - x_1 - x_2
                        num+=1
                        if all(x in [0, 1] for x in [x_0, x_1, x_2, x_3]):
                            if abs(taylor_num[num]) >= 0.01:
                                if x_0 == 1 and x_1 == 1:
                                    G.add_edge('x0', 'x1')
                                if x_0 == 1 and x_2 == 1:
                                    G.add_edge('x0', 'x2')
                                if x_0 == 1 and x_3 == 1:
                                    G.add_edge('x0', 'x3')
                                if x_1 == 1 and x_2 == 1:
                                    G.add_edge('x1', 'x2')
                                if x_1 == 1 and x_3 == 1:
                                    G.add_edge('x1', 'x3')
                                if x_2 == 1 and x_3 == 1:
                                    G.add_edge('x2', 'x3')
        if X.shape[1] == 5:
            G.add_nodes_from(['x0', 'x1', 'x2', 'x3', 'x4'])

            # 添加连接关系
            num = 0
            for x_0 in range(2):
                for x_1 in range(2):
                    for x_2 in range(2):
                        for x_3 in range(2):
                            x_4 = k - x_0 - x_1 - x_2 - x_3
                            num+=1
                            if all(x in [0, 1] for x in [x_0, x_1, x_2, x_3, x_4]):
                                if abs(taylor_num[num]) >= 0.01:
                                    if x_0 == 1 and x_1 == 1:
                                        G.add_edge('x0', 'x1')
                                    if x_0 == 1 and x_2 == 1:
                                        G.add_edge('x0', 'x2')
                                    if x_0 == 1 and x_3 == 1:
                                        G.add_edge('x0', 'x3')
                                    if x_0 == 1 and x_4 == 1:
                                        G.add_edge('x0', 'x4')
                                    if x_1 == 1 and x_2 == 1:
                                        G.add_edge('x1', 'x2')
                                    if x_1 == 1 and x_3 == 1:
                                        G.add_edge('x1', 'x3')
                                    if x_1 == 1 and x_4 == 1:
                                        G.add_edge('x1', 'x4')
                                    if x_2 == 1 and x_3 == 1:
                                        G.add_edge('x2', 'x3')
                                    if x_2 == 1 and x_4 == 1:
                                        G.add_edge('x2', 'x4')
                                    if x_3 == 1 and x_4 == 1:
                                        G.add_edge('x3', 'x4')
        if X.shape[1] == 6:
            G.add_nodes_from(['x0', 'x1', 'x2', 'x3', 'x4', 'x5'])

            for x_0 in range(2):
                for x_1 in range(2):
                    for x_2 in range(2):
                        for x_3 in range(2):
                            for x_4 in range(2):
                                x_5 = k - x_0 - x_1 - x_2 - x_3 - x_4
                                if all(x in [0, 1] for x in [x_0, x_1, x_2, x_3, x_4, x_5]):
                                    if abs(taylor_num[num]) >= 0.01:
                                        connections = [('x0', 'x1'), ('x0', 'x2'), ('x0', 'x3'), ('x0', 'x4'), ('x0', 'x5'),
                                                    ('x1', 'x2'), ('x1', 'x3'), ('x1', 'x4'), ('x1', 'x5'),
                                                    ('x2', 'x3'), ('x2', 'x4'), ('x2', 'x5'),
                                                    ('x3', 'x4'), ('x3', 'x5'),
                                                    ('x4', 'x5')]
                                        for connection in connections:
                                            if eval(f'x_{connection[0][1]}') == 1 and eval(f'x_{connection[1][1]}') == 1:
                                                G.add_edge(connection[0], connection[1])


        if X.shape[1] == 7:
            G.add_nodes_from(['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6'])
            # 添加连接关系
            for x_0 in range(2):
                for x_1 in range(2):
                    for x_2 in range(2):
                        for x_3 in range(2):
                            for x_4 in range(2):
                                for x_5 in range(2):
                                    x_6 = k - x_0 - x_1 - x_2 - x_3 - x_4 - x_5
                                    if all(x in [0, 1] for x in [x_0, x_1, x_2, x_3, x_4, x_5, x_6]):
                                        if abs(taylor_num[num]) >= 0.01:
                                            connections = [('x0', 'x1'), ('x0', 'x2'), ('x0', 'x3'), ('x0', 'x4'),
                                                           ('x0', 'x5'), ('x0', 'x6'),
                                                           ('x1', 'x2'), ('x1', 'x3'), ('x1', 'x4'), ('x1', 'x5'),
                                                           ('x1', 'x6'),
                                                           ('x2', 'x3'), ('x2', 'x4'), ('x2', 'x5'), ('x2', 'x6'),
                                                           ('x3', 'x4'), ('x3', 'x5'), ('x3', 'x6'),
                                                           ('x4', 'x5'), ('x4', 'x6'),
                                                           ('x5', 'x6')]
                                            for connection in connections:
                                                if eval(f'x_{connection[0][1]}') == 1 and eval(
                                                        f'x_{connection[1][1]}') == 1:
                                                    G.add_edge(connection[0], connection[1])
                                            
        if X.shape[1] == 8:
            G.add_nodes_from(['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7'])
            connections = [('x0', 'x1'), ('x0', 'x2'), ('x0', 'x3'), ('x0', 'x4'), ('x0', 'x5'), ('x0', 'x6'),('x0', 'x7'),
                           ('x1', 'x2'), ('x1', 'x3'), ('x1', 'x4'), ('x1', 'x5'), ('x1', 'x6'), ('x1', 'x7'),
                           ('x2', 'x3'), ('x2', 'x4'), ('x2', 'x5'), ('x2', 'x6'), ('x2', 'x7'),
                           ('x3', 'x4'), ('x3', 'x5'), ('x3', 'x6'), ('x3', 'x7'),
                           ('x4', 'x5'), ('x4', 'x6'), ('x4', 'x7'),
                           ('x5', 'x6'), ('x5', 'x7'),
                           ('x6', 'x7')]
            # 添加连接关系
            for x_0 in range(2):
                for x_1 in range(2):
                    for x_2 in range(2):
                        for x_3 in range(2):
                            for x_4 in range(2):
                                for x_5 in range(2):
                                    for x_6 in range(2):
                                        x_7 = k - x_0 - x_1 - x_2 - x_3 - x_4 - x_5 - x_6
                                        if all(x in [0, 1] for x in [x_0, x_1, x_2, x_3, x_4, x_5, x_6, x_7]):
                                            if abs(taylor_num[num]) >= 0.01:
                                                for connection in connections:
                                                    if eval(f'x_{connection[0][1]}') == 1 and eval(
                                                            f'x_{connection[1][1]}') == 1:
                                                        G.add_edge(connection[0], connection[1])
        if X.shape[1] == 9:
            G.add_nodes_from(['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7','x8'])
            connections = [('x0', 'x1'), ('x0', 'x2'), ('x0', 'x3'), ('x0', 'x4'),
                            ('x0', 'x5'), ('x0', 'x6'), ('x0', 'x7'), ('x0', 'x8'),
                            ('x1', 'x2'), ('x1', 'x3'), ('x1', 'x4'), ('x1', 'x5'),
                            ('x1', 'x6'), ('x1', 'x7'), ('x1', 'x8'),
                            ('x2', 'x3'), ('x2', 'x4'), ('x2', 'x5'), ('x2', 'x6'),
                            ('x2', 'x7'), ('x2', 'x8'),
                            ('x3', 'x4'), ('x3', 'x5'), ('x3', 'x6'), ('x3', 'x7'),
                            ('x3', 'x8'),
                            ('x4', 'x5'), ('x4', 'x6'), ('x4', 'x7'), ('x4', 'x8'),
                            ('x5', 'x6'), ('x5', 'x7'), ('x5', 'x8'),
                            ('x6', 'x7'), ('x6', 'x8'),
                            ('x7', 'x8')]
            # 添加连接关系
            for x_0 in range(2):
                for x_1 in range(2):
                    for x_2 in range(2):
                        for x_3 in range(2):
                            for x_4 in range(2):
                                for x_5 in range(2):
                                    for x_6 in range(2):
                                        for x_7 in range(2):
                                            x_8 = k - x_0 - x_1 - x_2 - x_3 - x_4 - x_5 - x_6 - x_7
                                            if all(x in [0, 1] for x in [x_0, x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8]):
                                                if abs(taylor_num[num]) >= 0.01:
                                                    for connection in connections:
                                                        if eval(f'x_{connection[0][1]}') == 1 and eval(
                                                                f'x_{connection[1][1]}') == 1:
                                                            G.add_edge(connection[0], connection[1])
        if X.shape[1] == 10:
            G.add_nodes_from(['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7','x8','x9'])
            connections = [('x0', 'x1'), ('x0', 'x2'), ('x0', 'x3'), ('x0', 'x4'),
                            ('x0', 'x5'), ('x0', 'x6'), ('x0', 'x7'), ('x0', 'x8'),
                            ('x0', 'x9'),
                            ('x1', 'x2'), ('x1', 'x3'), ('x1', 'x4'), ('x1', 'x5'),
                            ('x1', 'x6'), ('x1', 'x7'), ('x1', 'x8'), ('x1', 'x9'),
                            ('x2', 'x3'), ('x2', 'x4'), ('x2', 'x5'), ('x2', 'x6'),
                            ('x2', 'x7'), ('x2', 'x8'), ('x2', 'x9'),
                            ('x3', 'x4'), ('x3', 'x5'), ('x3', 'x6'), ('x3', 'x7'),
                            ('x3', 'x8'), ('x3', 'x9'),
                            ('x4', 'x5'), ('x4', 'x6'), ('x4', 'x7'), ('x4', 'x8'),
                            ('x4', 'x9'),
                            ('x5', 'x6'), ('x5', 'x7'), ('x5', 'x8'), ('x5', 'x9'),
                            ('x6', 'x7'), ('x6', 'x8'), ('x6', 'x9'),
                            ('x7', 'x8'), ('x7', 'x9'),
                            ('x8', 'x9')]

            # 添加连接关系
            for x_0 in range(2):
                for x_1 in range(2):
                    for x_2 in range(2):
                        for x_3 in range(2):
                            for x_4 in range(2):
                                for x_5 in range(2):
                                    for x_6 in range(2):
                                        for x_7 in range(2):
                                            for x_8 in range(2):
                                                x_9 = k - x_0 - x_1 - x_2 - x_3 - x_4 - x_5 - x_6 - x_7 - x_8
                                                if all(x in [0, 1] for x in [x_0, x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9]):
                                                    if abs(taylor_num[num]) >= 0.01:
                                                        for connection in connections:
                                                            if eval(f'x_{connection[0][1]}') == 1 and eval(
                                                                    f'x_{connection[1][1]}') == 1:
                                                                G.add_edge(connection[0], connection[1])

    subgraphs = list(nx.connected_components(G))
    num_subgraphs = len(subgraphs)

    print("图的节点集合:", G.nodes)



    # # 划分子式
    for i, subgraph in enumerate(subgraphs):
        x_variables =  [x for x in _x if str(x) in subgraph]
        print(f"子图{i+1}的变量集合：{x_variables}")
        print(len(x_variables))
        print(x_variables[0])
        # print(x_variables[1])
        if f_taylor == None:
                f_taylor = self.f_taylor

        # print(f"子图{i+1}的方程为：{f_taylor}")




        f= sympify(f_taylor)

        # 分割表达式为多个项
        terms = f.as_ordered_terms()

        # 打印分割后的各项
        for term in terms:
            print(term)

        print(f"子图{i+1}的方程为：{f}")
        metric.varNum =2

        y_pred = metric._calY(f, _x)

        nmse = mean_squared_error(Y, y_pred)

    print(f"子图{i+1}XXX的方程为：{f}")
    print(f"子图{i+1}的方程的NMSE为：{nmse}")



    # 生成新的方程
    #对于这样的项-1.0866452424638696e-5*x0**2我们首先要判断其系数是否小于0.01,如果小于0.01则将其舍弃
    f_new = filter_equation(str(f))

    print(f"子图{i+1}X的方程为：{f_new}")

    terms = split_formula(str(f_new))

    print(f"子图{i+1}的方程的项数为：{len(terms)}")
    print(f"子图{i+1}的方程的项为：{terms}")

    equations = assign_terms_to_equations(terms, subgraphs)
    print(f"Equations: {equations}")

    func = []
    y_pred = []
    # for i, subgraph in enumerate(subgraphs):
    #     x_variables = [x for x in _x if str(x) in subgraph]

    print(f"子图的变量集合：{x_variables}")

    for i, subgraph in enumerate(subgraphs):
        equations_temp = copy.deepcopy(equations)
        print(f"equations_temp",equations_temp)
        x_variables =  [x for x in _x if str(x) in subgraph]
        print(f"子图{i+1}WW的变量集合：{x_variables}")
        # x_variables =  [x for x in _x if str(x) in subgraphs]

        # print(f"子图{i+1}Q的变量集合：{x_variables}")


        if equations_temp.split('\n')[i] !="NONE":
            equation_str = equations_temp.split('\n')[i]
        else:
            print("yyyy")
            continue


        # vars = symbols(' '.join(subgraphs[i]))
        print(f"equation_str: {equation_str}")
        func.append(sympify(equation_str))
        # func= sympify(equation_str)
        print(func[i])

        
        # print(f"子图{i+1}的方程为：{func}")

        # print(f"x_variables: {x_variables[i]}")

        result = metric._calY(func[i], _x)  # Remove unnecessary indexing
        if isinstance(result, list):
            y_pred.extend(result)  # Extend the list if result is a list
        else:
            y_pred.append(result)  # Append to the list if result is not a list

        # print(f"y_pred: {y_pred}")

        # nmse = mean_squared_error(Y,y_pred)
        # print(f"子图{i+1}的方程的NMSE为：{nmse}")

        # end_fitness, programs,population,Y_pred= Taylor_Based_SR(x_variables,X,metric.change_Y(Y),qualified_list,eq_write,population,Generation,Pop,repeatNum,low_polynomial)
        end_fitness, programs,population,Y_pred= Taylor_Based_SR(_x,X,change_Y(Y),qualified_list,eq_write,Pop,Generation,repeatNum,qualified_list[2] < 1e-5, SR_method=SR_method)
        end_fitness, programs, population, Y_pred = Taylor_Based_SR(_x, X, change_Y(Y, qualified_list), qualified_list,
                                                                    eq_write, population, Generation, Pop, repeatNum,
                                                                    qualified_list[2] < 1e-5, SR_method=SR_method)
        # print(f"子图{i+1}的方程的fitness为：{end_fitness}")
        # print(programs.get_expression())


        
    if num_subgraphs == 1:
        print("图是连通的,没有划分子图！")
        return False
    
    else:
        print("图是不连通的,划分子图如下：")
        for i, subgraph in enumerate(subgraphs):
            print(f"子图{i+1}的节点集合：{subgraph}")

        return True

def CalTaylorFeatures(f_taylor, _x, X, Y, population, Generation, Pop, repeatNum, eq_write):
    print('In CalTaylorFeatures')
    metric = Metrics2(f_taylor, _x, X, Y)
    if metric.judge_Low_polynomial():
        return [metric.low_nmse], [metric.f_low_taylor], None
    if X.shape[1] > 1:
        if metric.judge_additi_separability():
            print('Separability of addition')
            print('===========================start left recursion============================')
            low_mse1, f_add1, population1 = CalTaylorFeatures(metric.f_left_taylor, metric._x_left, metric.X_left,
                                                              metric.Y_left, population, Generation, Pop // 2,
                                                              repeatNum, eq_write)
            low_mse1 = low_mse1[0]
            f_add1 = f_add1[0]
            print('===========================start right recursion============================')
            low_mse2, f_add2, population2 = CalTaylorFeatures(metric.f_right_taylor, metric._x_right, metric.X_right,
                                                              metric.Y_right, population, Generation, Pop // 2,
                                                              repeatNum, eq_write)
            # if population2 != None:
            low_mse2 = low_mse2[0]
            f_add2 = f_add2[0]
            # print("f_add1: ",f_add1," f_add2: ",f_add2)
            f_add = sympify(str(f_add1) + '+' + str(f_add2))
            try:
                y_pred_add = metric._calY(f_add, _x, metric._X)
                nmse = mean_squared_error(Y, y_pred_add)
                if nmse < metric.low_nmse:
                    return [nmse], [f_add], None
                else:
                    return [metric.low_nmse], [metric.f_low_taylor], None
            except BaseException:
                return [metric.low_nmse], [metric.f_low_taylor], None
        elif metric.judge_multi_separability():
            print('multiplicative separability')
            print('===========================start left recursion============================')
            low_mse1, f_multi1, population1 = CalTaylorFeatures(metric.f_left_taylor, metric._x_left, metric.X_left,
                                                                metric.Y_left, population, Generation, population,
                                                                Pop // 2, repeatNum, eq_write)
            # if population1 != None:
            low_mse1 = low_mse1[0]
            f_multi1 = f_multi1[0]
            print('===========================start right recursion============================')
            low_mse2, f_multi2, population2 = CalTaylorFeatures(metric.f_right_taylor, metric._x_right, metric.X_right,
                                                                metric.Y_right, population, Generation, Pop // 2,
                                                                repeatNum, eq_write)
            # if population2 != None:
            low_mse2 = low_mse2[0]
            f_multi2 = f_multi2[0]
            f_multi = sympify('(' + str(f_multi1) + ')*(' + str(f_multi2) + ')')
            try:
                y_pred_multi = metric._calY(f_multi, _x, metric._X)
                nmse = mean_squared_error(Y, y_pred_multi)
                if nmse < metric.low_nmse:
                    return [nmse], [f_multi], None
                else:
                    return [metric.low_nmse], [metric.f_low_taylor], None
            except BaseException:
                return [metric.low_nmse], [metric.f_low_taylor], None

    qualified_list = []
    qualified_list.extend(
        [metric.judge_Bound(), metric.f_low_taylor, metric.low_nmse, metric.bias, metric.judge_parity(),
         metric.judge_monotonicity()])
    return Taylor_Based_SR(_x, X, metric.change_Y(Y), qualified_list, eq_write, population, Generation, Pop, repeatNum,
                           metric.judge_Low_polynomial())


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
    _init()
    x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24, x25, x26, x27, x28, x29 = symbols(
        "x0,x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16,x17,x18,x19,x20,x21,x22,x23,x24,x25,x26,x27,x28,x29 ")

    set_value('_x',
              [x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22,
               x23,
               x24, x25, x26, x27, x28, x29])
    _x = get_value('_x')
    eqName = 'SubTaylorGP.out'  # eqName = fileName[:-4]+'.out' 原先的打印都是针对整体的，所以此处应该都用不上了
    eq_write = open(eqName, "w+")
    eq_write.write(
        'Gen|Top1|Length1|Fitness1|Top2|Length2|Fitness2|Top3|Length3|Fitness3|Top4|Length4|Fitness4|Top5|Length5|Fitness5|Top6|Length6|Fitness6|Top7|Length7|Fitness7|Top8|Length8|Fitness8|Top9|Length9|Fitness9|Top10|Length10|Fitness10\n')

    X, Y = np.split(X_Y, (-1,), axis=1)
    end_fitness, program, findBestFlag, Y_pred, temp_Y_pred = None, None, False, None, copy.deepcopy(Y_pred)
    if qualified_list == None:  # 保证泰勒多项式不用重复计算
        Metric = []
        for linalg in ["solve", "lstsq"]:
            loopNum = 0
            while True:
                metric = Metrics(varNum=X.shape[1], dataSet=X_Y, linalg=linalg)
                loopNum += 1
                Metric.append(metric)
                if loopNum == 5 and X.shape[1] <= 2:
                    break
                elif loopNum == 4 and (X.shape[1] > 2 and X.shape[1] <= 3):
                    break
                elif loopNum == 3 and (X.shape[1] > 3 and X.shape[1] <= 4):
                    break
                elif loopNum == 2 and (X.shape[1] > 4):
                    break
        Metric.sort(key=lambda x: x.low_nmse)
        metric = Metric[0]
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
            print(qualified_list)

    # elif metric.nihe_flag and (metric.judge_additi_separability() or metric.judge_multi_separability()):
    #     end_fitness, programs,population = CalTaylorFeatures(metric.f_taylor, _x[:X.shape[1]], X, Y, population,Generation,Pop, repeatNum, eq_write)
    if end_fitness is None:

        end_fitness,programs,population,Y_pred= Taylor_Based_SR(_x,X,change_Y(Y),qualified_list,
                                                                eq_write,population,Generation,
                                                                Pop,repeatNum,
                                                                qualified_list[2] < 1e-5, SR_method=SR_method)
        A=CalTaylorFeaturesGraph(metric.taylor_num,metric.f_taylor,
                            #    _x[:X.shape[1]],
                               _x,
                               X,Y,population,Generation,
                               Pop,repeatNum,
                               eq_write,qualified_list,qualified_list[2] <1e-5,SR_method=SR_method)
        # end_fitness, programs, population, Y_pred = Taylor_Based_SR(_x, X, change_Y(Y, qualified_list), qualified_list,
        #                                                             eq_write, population, Generation, Pop, repeatNum,
        #                                                             qualified_list[2] < 1e-5, SR_method=SR_method)
        if isinstance(Y_pred,str):
            Y_pred = temp_Y_pred
        # Y_pred = programs[0].predict(X)
        print(end_fitness)
        print(programs)
        print('fitness_and_program', end_fitness[0], programs[0], sep='\n')
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
