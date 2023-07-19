from gplearn.genetic import SymbolicRegressor
from keplar.data.data import Data
from keplar.operator.evaluator import SingleBingoEvaluator
from keplar.translator.translator import gp_to_bingo, gp_to_bingo1

data = Data("pmlb", "1027_ESL", ["x1", "x2", "x3", 'y'])
data.read_file()
x = data.get_np_x()
y = data.get_np_y()
sr = SymbolicRegressor(population_size=128, generations=1000,
                       function_set=["add", "sub", "mul", "div", "sin", "cos", 'sin'])
tt = sr.fit(X=x, y=y)
print(str(tt))
aa = gp_to_bingo1(str(tt))
eval = SingleBingoEvaluator(data, equation=str(aa))
print(eval.do())
