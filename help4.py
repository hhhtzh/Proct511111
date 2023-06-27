import re

import numpy as np
import pandas as pd
import pyoperon as Operon
import conversion
from bingo.symbolic_regression import AGraph
from keplar.data.data import Data
from keplar.operator.feature_engineering import FeatureEngineering
from keplar.population.individual import Individual
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
from keplar.translator.translator import prefix_to_postfix, bingo_infixstr_to_func

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
