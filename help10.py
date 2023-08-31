import pandas as pd

from keplar.data.data import Data
from keplar.operator.evaluator import OperonSingleEvaluator

# data = Data("pmlb", "1027_ESL", ["x0", "x1", "x2", "x3", 'y'])
# data.read_file()
# x = data.get_np_x()
# y = data.get_np_y()
#
# eval = OperonSingleEvaluator("RMSE", x, y, 0.7, True, "X_0")
# print(eval.do())
data = pd.read_csv("NAStraining_data/recursion_training2.csv")
# print(data)
data1 = data.iloc[:, :-2]
print(data1)
