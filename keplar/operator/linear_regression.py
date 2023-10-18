import re

import numpy as np

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
        new_expression = expression[4:]
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


class SklearnTwoIndividualLinearRegression(LinearRegression):
    def __init__(self, data):
        super().__init__()
        self.data = data

    def do(self, population):
        model = sklr()
        old_x = self.data.get_np_x()
        y = self.data.get_np_y()
        [parent_1_num, parent_2_num] = np.random.randint(low=0, high=population.get_pop_size() - 1, size=2)
        parent_1 = population.pop_list[parent_1_num]
        parent_2 = population.pop_list[parent_2_num]
        equ1 = str(parent_1.format())
        equ2 = str(parent_2.format())
        equ1 = re.sub(r"power", "^", equ1)
        equ2 = re.sub(r"power", "^", equ2)
        equ1 = re.sub(r" pow ", " ^ ", equ1)
        equ2 = re.sub(r" pow ", " ^ ", equ2)
        bingo_parent_1 = AGraph(equation=equ1)
        bingo_parent_2 = AGraph(equation=equ2)
        bingo_parent_1._update()
        bingo_parent_2._update()
        print(str(bingo_parent_1))
        print(str(bingo_parent_2))
        print(np.shape(old_x))
        y1 = bingo_parent_1.evaluate_equation_at(old_x)
        y2 = bingo_parent_2.evaluate_equation_at(old_x)
        y1 = np.array(y1).reshape(-1, 1)
        y2 = np.array(y2).reshape(-1, 1)
        temp_x = [y1, y2]
        x = np.array(temp_x).reshape(-1, 2)
        print(x)
        print(y)
        model.fit(x, y)
        # 获取截距和系数
        intercept = model.intercept_
        coefficients = model.coef_
        # 打印截距和系数
        print("截距 (b0):", intercept)
        print("系数 (b1, b2):", coefficients)
        equ_list = [str(bingo_parent_1), str(bingo_parent_2)]
        # 将模型表示为表达式
        expression = f"y = {intercept} + "
        for i, coef in enumerate(coefficients):
            expression += f"{coef} * ({equ_list[i - 1]}) + "
        expression = expression[:-3]  # 去除末尾的 " + "
        print("模型表达式:", expression)
        print(expression[4:])
        new_expression = expression[4:]
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


class SklearnOneIndividualLinearRegression(LinearRegression):
    def __init__(self, data):
        super().__init__()
        self.data = data

    def do(self, population):
        model = sklr()
        old_x = self.data.get_np_x()
        y = self.data.get_np_y()
        parent_num=population.get_best()
        parent_= population.pop_list[parent_num]
        equ1 = str(parent_.format())
        equ1 = re.sub(r"power", "^", equ1)
        equ2 = re.sub(r"power", "^", equ2)
        equ1 = re.sub(r" pow ", " ^ ", equ1)
        equ2 = re.sub(r" pow ", " ^ ", equ2)
        bingo_parent_1 = AGraph(equation=equ1)
        bingo_parent_1._update()
        y1 = bingo_parent_1.evaluate_equation_at(old_x)
        y1 = np.array(y1).reshape(-1, 1)
        x=y1
        print(x)
        print(y)
        model.fit(x, y)
        # 获取截距和系数
        intercept = model.intercept_
        coefficients = model.coef_
        # 打印截距和系数
        print("截距 (b0):", intercept)
        print("系数 (b1, b2):", coefficients)
        equ_list = [str(bingo_parent_1)]
        # 将模型表示为表达式
        expression = f"y = {intercept} + "
        for i, coef in enumerate(coefficients):
            expression += f"{coef} * ({equ_list[i - 1]}) + "
        expression = expression[:-3]  # 去除末尾的 " + "
        print("模型表达式:", expression)
        print(expression[4:])
        new_expression = expression[4:]
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
