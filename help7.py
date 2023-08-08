import pandas as pd

from bingo.symbolic_regression.agraph.string_parsing import eq_string_to_infix_tokens, infix_to_postfix
from gplearn._program import _Program
from keplar.Algorithm.Alg import BingoAlg
from keplar.data.data import Data
from keplar.operator.statistic import BingoStatistic
from keplar.translator.translator import bgpostfix_to_gpprefix, bingo_to_gp

# data = Data("pmlb", "1027_ESL", ["x1", "x2", "x3", 'y'])
data = Data("txt", "datasets/vla/two/1.txt", ["x0", "x1", "y"])
# data = Data("txt", "datasets/2.txt", ["x0", "x1", "x2", "x3", "x4", "y"])
data.read_file()
data.set_xy("y")
fit_list = []
time_list = []
bingo = BingoAlg(10, data,
                 operators=["+", "-", "*", "/", "sin", "exp", "cos", 'sqrt', 'log', 'sin', 'pow', 'exp', '^'])
bingo.run()
bg_graph=bingo.island.get_best_individual()
print(bg_graph)
tk=bingo_to_gp(str(bg_graph))
tk = eq_string_to_infix_tokens(str(tk))
print(tk)
tk = infix_to_postfix(tk)
print(tk)
tk = bgpostfix_to_gpprefix(tk)
print(tk)
gp_prog = _Program(function_set=["add", "sub", "mul", "div", "sqrt","neg","power"],
                   arities={"add": 2, "sub": 2, "mul": 2, "div": 2, "sqrt": 1,"neg":1,"power":2},
                   init_depth=[3, 3], init_method="half and half", n_features=4, const_range=[0, 1], metric="rmse",
                   p_point_replace=0.4, parsimony_coefficient=0.01, random_state=1, program=tk)

# for i in range(10):