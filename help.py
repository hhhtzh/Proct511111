import re

import numpy as np
import pandas
from pmlb import fetch_data
import pyoperon as Operon
from bingo.evaluation.evaluation import Evaluation
from bingo.local_optimizers.local_opt_fitness import LocalOptFitnessFunction
from bingo.local_optimizers.scipy_optimizer import ScipyOptimizer
from gplearn.genetic import SymbolicRegressor

from bingo.symbolic_regression import AGraph, AGraphCrossover, ExplicitTrainingData, ExplicitRegression
from keplar.data.data import Data
from keplar.operator.creator import GpCreator, BingoCreator, OperonCreator
from keplar.population.individual import Individual
from keplar.translator.translator import to_op, trans_op

# bingo_parent_1 = AGraph(use_simplification=True,equation="(X_0 + X_0 - (X_0 + X_0) - (X_0) - (X_0))/(X_0 + X_0 - (X_0 + X_0) - (X_0)) - ((X_0 + X_0 - (X_0 + X_0) - (X_0))/(X_0 + X_0 + X_0 - (X_0 + X_0) - (X_0)))")
# bingo_parent_2 = AGraph(use_simplification=True,equation="(X_0 + X_0 - (X_0 + X_0) - (X_0) - (X_0))/(X_0 + X_0 - (X_0 + X_0) - (X_0)) - ((X_0 + X_0 - (X_0 + X_0) - (X_0))/(X_0 + X_0 + X_0 - (X_0 + X_0) - (X_0)))")
# bingo_parent_1._update()
# bingo_parent_2._update()
# crossover = AGraphCrossover()
# bingo_child_1, bingo_child_2 = crossover(parent_1=bingo_parent_1, parent_2=bingo_parent_2)
x, y = fetch_data('1027_ESL', return_X_y=True, local_cache_dir='./datasets')
# reg=SymbolicRegressor(generations=1,population_size=1)
# reg.fit(x,y)
# print(reg)
# gpc = BingoCreator(50, ["+", "-", "*", "/"], x, 10)
# gpc.do()
# gpc=GpCreator(50,x,y)
# gpc.do()
# dt = pandas.DataFrame([
#     [1, 2, 3],
#     [1, 1, 1],
#     [1, 2, 3]
# ])
# dt.head()
# dt = Data("pmlb", "1027_ESL")
# dt.read_file()
# dt.display_data()
# def postfix_to_infix(expression):
#     stack = []
#
#     for token in expression:
#         if token.isalnum():  # 操作数，直接入栈
#             stack.append(token)
#         else:  # 运算符，弹出两个操作数并生成中缀表达式
#             operand2 = stack.pop()
#             operand1 = stack.pop()
#             infix = "(" + operand1 + token + operand2 + ")"
#             stack.append(infix)
#
#     return stack.pop()  # 返回最终中缀表达式
#
#
# # 示例用法
# postfix_expression = "ab+c*"
# infix_expression = postfix_to_infix(postfix_expression)
# print(infix_expression)
# x=np.array(x)
# y=np.array(y).reshape([-1,1])
# print(x.shape,y.shape)
# opc = OperonCreator("balanced", x, y, 128)
# opc.do()
D = fetch_data('1027_ESL', return_X_y=False, local_cache_dir='./datasets').to_numpy()

# initialize a dataset from a numpy array
ds = Operon.Dataset(D)

x1=ds.GetVariable("X1")
nd1 = Operon.Node.Constant(11)
nd2 = Operon.Node.Variable(1)
nd2.HashValue=2673900822390960787
nd3 = Operon.Node.Div()
nd_list = [nd1,nd2, nd3]
tr = Operon.Tree(nd_list)
print(Operon.InfixFormatter.Format(tr, ds, 5))
variables=ds.Variables
for var in variables:
    print(var.Hash,var.Name)
print(x1.Hash)
# print(ds.VariableNames)


# equ="(111.4449)*(X_2)+22.777+exp(4444444444)"
# ind=Individual(equation=equ)
# euq=to_op(ind)
# print("2".isalpha())
# bingo_ind=AGraph(equation=equ)
# bingo_ind._update()
# print(bingo_ind.command_array)
# training_data = ExplicitTrainingData(x, y)
# fitness = ExplicitRegression(training_data=training_data)
# optimizer = ScipyOptimizer(fitness, method="lm")
# local_opt_fitness = LocalOptFitnessFunction(fitness, optimizer)
# evaluator = Evaluation(local_opt_fitness)
# evaluator(population=[bingo_ind])
# equ="X231"
# equ1=re.sub(r'X(\d{3})', r'X_\1', equ)
# equ1=re.sub(r'X(\d{2})', r'X_\1', equ1)
# equ1=re.sub(r'X(\d{1})', r'X_\1', equ1)
# pattern = r'(X_\d+)'
# output_string = re.sub(pattern, lambda m: m.group(1)[:-1] + str(int(m.group(1)[-1]) - 1),equ1)
# print(output_string)
