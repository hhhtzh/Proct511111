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
# data = pd.read_csv("NAStraining_data/recursion_training2.csv")
# # print(data)
# data1 = data.iloc[:, :-2]
# print(data1)
import re

# 输入的字符串表达式
expression = "X_0 + X_1 + X_2"

# 使用正则表达式查找匹配 "X_a" 的部分，并进行替换
expression = re.sub(r'X_(\d+)', lambda m: f'X_{int(m.group(1)) - 1}', expression)

# 打印替换后的表达式
print(expression)

