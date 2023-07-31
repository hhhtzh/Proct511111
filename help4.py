import csv
import os
import re

import numpy as np
import pandas as pd
import pyoperon as Operon
import conversion
from numpy.random import RandomState
from sympy import sympify, symbols
from sympy import Integer

from bingo.symbolic_regression import AGraph
from bingo.symbolic_regression.agraph.operator_definitions import INTEGER, CONSTANT
from bingo.symbolic_regression.agraph.string_parsing import eq_string_to_infix_tokens, infix_to_postfix, operators, \
    functions, operator_map, var_or_const_pattern, int_pattern
from keplar.data.data import Data
from keplar.operator.creator import GpCreator
from keplar.operator.evaluator import SingleBingoEvaluator
from keplar.operator.statistic import BingoStatistic, TaylorStatistic
# from keplar.operator.feature_engineering import FeatureEngineering, TaylorFeature
from keplar.population.individual import Individual
from keplar.preoperator.ml.sklearndbscan import SklearnDBscan
# def is_operator(char):
#     operators = ['+', '-', '*', '/']
#     return char in operators
#
#
# def postfix_to_prefix(expression):
#     stack = []
#     for char in expression:
#         if is_operator(char):
#             operand2 = stack.pop()
#             operand1 = stack.pop()
#             stack.append(char + operand1 + operand2)
#         else:
#             stack.append(char)
#     return stack
#
#
# # 测试
# postfix_expression = ['2', '3', '4', '*', '+', '5', '-']
# prefix_expression = postfix_to_prefix(postfix_expression)
# print("前缀表达式：", prefix_expression)
# dict={"11":2}
# print(dict["11"])
# str_equ="(X_1)((X_1 - ((X_1)/(X_1)))((X_1)(X_1)) - (X_0 - (X_0)))"
from keplar.translator.translator import prefix_to_postfix, bingo_infixstr_to_func, trans_op1, trans_op2

# def prefix_to_postfix(expression):
#     stack = []
#     operators = {'add': 1, 'sub': 2, 'mul': 3, 'div': 4, 'sqrt': 5, 'log': 6, 'abs': 7,
#                  'neg': 8, 'inv': 9, 'max': 10, 'min': 11, 'sin': 12, 'cos': 13, 'tan': 14,
#                  'sig': 15, 'aq': 16, 'pow': 17, 'exp': 18, 'square': 19,}  # 可用的运算符
#
#     for token in reversed(expression):
#         if token in operators:  # 操作符
#             # 弹出栈顶运算符，直到遇到更低优先级的运算符或左括号
#             while stack and stack[-1] in operators and operators[token] <= operators[stack[-1]]:
#                 yield stack.pop()
#             stack.append(token)
#         elif token == ')':  # 右括号
#             stack.append(token)
#         elif token == '(':  # 左括号
#             # 弹出栈顶运算符，直到遇到右括号
#             while stack and stack[-1] != ')':
#                 yield stack.pop()
#             stack.pop()  # 弹出右括号
#         else:  # 操作数
#             yield token
#
#     # 弹出栈中剩余的运算符
#     while stack:
#         yield stack.pop()
#
#
# prefix_expression = ['add', 'mul', '5', '2', '4']
# postfix_expression = list(prefix_to_postfix(prefix_expression))
# print("前缀表达式:", prefix_expression)
# print("后缀表达式:", postfix_expression)
# array = [1001, 5001, 5002]
# ind = Individual(func=array)
# print(ind.format())
# bing_ind = AGraph(True, ind.format())
# x = "((X_2)/((X_1)/(X_1)))(X_1)"
# y = "* / X_2 / X_1 X_1 X_1"
# node = "1"
# if node.isalnum():
#     print("xx")

# span=Operon.MakeSpan(array)
# dataset = Data("txt", "datasets/1.txt", ["x", "y"])
# dataset.read_file()
# fea_en = FeatureEngineering(["PCA", "filter"], dataset)
# fea_en.do()
# Operon.MakeSpan().
# ind=Operon.Individual()
# const_list=Operon.IndividualCollection()
# sel=Operon.RandomSelector()
# print(type(const_list))
# ws=Operon.OffspringGeneratorBase.Prepare(const_list)
# rd=RandomState(100)
# print(rd.random(2))
data = Data("pmlb", "1027_ESL", ["x1", "x2", "x3", "x4", 'y'])
data.read_file()
# data.read_file()
# ty=TaylorFeature(data,"test1")
# ty.do()
# aa=np.array([
#     [1,2],
#     [2,3],
#     [4,5]
# ])
# print(aa.shape[1])
# # A=Operon.RMSE()
# data = Data("txt", "datasets/1.txt", ["x", "y"])
# # data = Data("txt", "datasets/2.txt", ["x0", "x1", "x2", "x3", "x4", "y"])
# data.read_file()
# data.set_xy("y")
# x = data.get_np_x()
# y = data.get_np_y()
# cr = GpCreator(100, x, y, "bingo", n_jobs=3)
# cr.do()
# a = [1, 2, 3]
# b=a.pop(1)
# print(b)
# print(a)
sk=SklearnDBscan()
ds = data.get_np_ds()
# print(ds[:, :-1])
end1=sk.do(data)
print(end1)
# for i in range(2):
#     print("ee")
# a = [1, 2, 3]
# a = np.array(a).reshape(1, -1)
# b = np.array([1, 1, 1]).reshape(1, -1)
# a = np.vstack((a, b))
# print(a)
# a.sort(reverse=True)
# print(a)
# a=1
# b=[0]*9
# print(b)
# a=[1,1,1]
#
# for b,i in enumerate(a):
#     print("b:"+str(b)) 
#     print("i:"+str(i))
# a = [[1], [1], [1]]
# a = np.array(a)
# print(np.shape(a))
# b = [[2], [1], [1]]
# c=np.append(a, b, axis=1)
# print(c)
# print(a.any() == 0)
# str1 = "0.6011560693641608*x0**3 + 0.7225433526011544*x0**2*x1 + 0.00226680267482582*x0**2*x2 - 12.15935622804032*x0**2 + 0.5693641618497095*x0*x1**2 - 0.1556160036268847*x0*x1*x2 - 12.55961691034792*x0*x1 + 0.2322906041029134*x0*x2**2 - 0.8638218293097488*x0*x2 + 0.20231213872832335*x0*x3 + 89.987192564886985*x0 + 0.5196118909991743*x1**3 + 0.06793324266122511*x1**2*x2 - 11.525448907886846*x1**2 + 0.26592428879066154*x1*x2**2 - 2.3880624504136827*x1*x2 + 0.6820809248554924*x1*x3 + 94.508856721879364*x1 - 0.08093902300804774*x2**3 - 1.891561827042956*x2**2 + 0.5260115606936419*x2*x3 + 15.67149778986735*x2 + 0.1556160036268847*x3**2 - 8.4063243794627774*x3 - 302.5455951166587"
# str3="(sqrt((X_2)(6.847460619528528 + (X_1)(X_3))))/(2.6348796890765738)"
# str2="-0.8749999999999968*x0**2 + 1.749999999999993*x0*x1 - 2.999999999999993*x0*x2 + 15.87499999999997*x0 + " \
#      "1.4583333333333247*x1**2 - 24.208333333333211*x1 + 17.99999999999996*x2 + 0.9999999999999982*x3 - " \
#      "34.000000000000105"
# str2 = re.sub(r'x(\d{1})', r'x_\1', str2)

# fit=eval.do()
# print(fit)
# x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10 = symbols('X_1 X_2 X_3 X_4 X_5 X_6 X_7 X_8 X_9 X_10')
# str2 = sympify(str1)
# print(str2)
# sta = BingoStatistic(str1)
#
# if not sta.pos_do():
#     print("ii")
# str1=trans_op2(str1)
# eval=SingleBingoEvaluator(data=data,equation=str1)
# fit=eval.do()
# print(fit)
# for cluster in range(0, 10):
#     print(cluster)

# dic = {'X_1': 0.444444}
# print(dic)
# print('X1' in dic)

