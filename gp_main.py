from gplearn.genetic import SymbolicRegressor
from keplar.data.data import Data

data = Data("pmlb", "1027_ESL", ["x1", "x2", "x3", 'y'])
data.read_file()
x=data.get_np_x()
y=data.get_np_y()
sr = SymbolicRegressor(population_size=128, generations=1000,
                       function_set=["add", "sub", "mul", "div", "sin", "exp", "cos", 'sqrt', 'log', 'sin', 'pow', 'exp', '^'])
sr.fit(X=x,y=y)
