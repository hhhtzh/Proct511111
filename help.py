import pandas
from pmlb import fetch_data

from gplearn.genetic import SymbolicRegressor

from bingo.symbolic_regression import AGraph, AGraphCrossover
from keplar.operator.creator import GpCreator, BingoCreator

# bingo_parent_1 = AGraph(use_simplification=True,equation="(X_0 + X_0 - (X_0 + X_0) - (X_0) - (X_0))/(X_0 + X_0 - (X_0 + X_0) - (X_0)) - ((X_0 + X_0 - (X_0 + X_0) - (X_0))/(X_0 + X_0 + X_0 - (X_0 + X_0) - (X_0)))")
# bingo_parent_2 = AGraph(use_simplification=True,equation="(X_0 + X_0 - (X_0 + X_0) - (X_0) - (X_0))/(X_0 + X_0 - (X_0 + X_0) - (X_0)) - ((X_0 + X_0 - (X_0 + X_0) - (X_0))/(X_0 + X_0 + X_0 - (X_0 + X_0) - (X_0)))")
# bingo_parent_1._update()
# bingo_parent_2._update()
# crossover = AGraphCrossover()
# bingo_child_1, bingo_child_2 = crossover(parent_1=bingo_parent_1, parent_2=bingo_parent_2)
# x, y = fetch_data('1027_ESL', return_X_y=True, local_cache_dir='./datasets')
# reg=SymbolicRegressor(generations=1,population_size=1)
# reg.fit(x,y)
# print(reg)
# gpc = BingoCreator(50, ["+", "-", "*", "/"], x, 10)
# gpc.do()
# gpc=GpCreator(50,x,y)
# gpc.do()
dt = pandas.DataFrame([
    [1, 2, 3],
    [1, 1, 1],
    [1, 2, 3]
])
x = dt.columns[2:3]
print(dt[x])
