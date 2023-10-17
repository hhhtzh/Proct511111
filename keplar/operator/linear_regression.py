import re

from bingo.symbolic_regression import AGraph
from keplar.operator.operator import Operator
from sklearn.linear_model import LinearRegression as sklr

from keplar.population.individual import Individual
from keplar.translator.translator import bingo_infixstr_to_func


class LinearRegression(Operator):
    def __init__(self):
        super().__init__()


class SklearnLinearRegression(LinearRegression):
    def __init__(self, data):
        super().__init__()
        self.data = data

    def do(self, population):
        model = sklr()
        x = self.data.get_np_x()
        y = self.data.get_np_y()
        print(x)
        print(y)
        model.fit(x, y)
        # 获取截距和系数
        intercept = model.intercept_
        coefficients = model.coef_

        # 打印截距和系数
        print("截距 (b0):", intercept)
        print("系数 (b1, b2):", coefficients)

        # 将模型表示为表达式
        expression = f"y = {intercept} + "
        for i, coef in enumerate(coefficients):
            expression += f"{coef} * x{i + 1} + "
        expression = expression[:-3]  # 去除末尾的 " + "
        print("模型表达式:", expression)
        print(expression[4:])
        new_expression=expression[4:]
        new_expression = re.sub(r'x(\d{3})', r'X_\1', new_expression)
        new_expression = re.sub(r'x(\d{2})', r'X_\1', new_expression)
        new_expression = re.sub(r'x(\d{1})', r'X_\1', new_expression)
        if population.pop_type != 'self':
            raise ValueError("种群类型必须为Keplar")
        else:
            bingo_ind = AGraph(equation=new_expression)
            bingo_ind._update()
            kep_func, const_arr = bingo_infixstr_to_func(str(bingo_ind))
            kep_ind = Individual(kep_func, const_arr)
            population.pop_list.append(kep_ind)
            population.pop_size += 1
